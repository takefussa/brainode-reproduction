"""Run resumable patient mesh sequences on explicitly specified GPUs.

Code and the observation table are frozen under the home project before any
workers start. Quality flags retain candidates and never change eligibility.
"""

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import signal
import sys
import time
import traceback

from mesh_cohort import ORDERING_POLICY, QC_THRESHOLDS, ordered_scans, subject_folder

PROJECT = Path(__file__).resolve().parent
SNAPSHOT_FILES = ['mesh_fit.py', 'mesh_input.py', 'mesh_initialization.py', 'mesh_cohort.py',
                  'mesh_batch.py', 'dataset/Shapes/hippocampus_template_mesh.obj',
                  'references/BrainSubcorticalStructure/models/pointnet.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def freeze_sources(directory, table):
    if not directory.resolve().is_relative_to(PROJECT):
        raise ValueError('code snapshot must remain under the home project')
    directory.mkdir(parents=True, exist_ok=True)
    paths = [(PROJECT/name, directory/name) for name in SNAPSHOT_FILES]
    paths.append((table, directory/'eligible.csv'))
    hashes = {}
    for source, target in paths:
        if target.exists():
            if digest(source) != digest(target):
                raise ValueError(f'existing snapshot differs: {target}; create a new snapshot')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        hashes[str(target.relative_to(directory))] = digest(target)
    write_json(directory/'snapshot.json', dict(files=hashes, source_project=str(PROJECT),
                                              source_table=str(table.resolve())))
    return directory/'eligible.csv'


def worker(device, tasks, events, settings):
    try:
        os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        sys.path.insert(0, settings['snapshot'])
        sys.modules.pop('mesh_cohort', None)
        # Spawned workers have not imported mesh_fit or its local dependencies.
        import mesh_fit
        import torch
        torch.cuda.set_device(device)
    except Exception:
        events.put(dict(event='fatal_worker', device=device, error=traceback.format_exc()))
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        subject, directory = task
        directory = Path(directory)
        log = Path(settings['output'])/'logs'/f'{subject_folder(subject)}.log'
        events.put(dict(event='started', subject=subject, device=device))
        try:
            args = ['--subject', subject, '--scan-table', settings['table'], '--output', str(directory),
                    '--device', f'cuda:{device}', '--iterations', str(settings['iterations']),
                    '--seed', str(settings['seed']), '--scope', settings['scope'],
                    '--update-mode', 'fixed-source', '--edge-mode', 'public',
                    '--template-initialization', 'mask-volume']
            cache = Path(settings['cache_root'])/subject_folder(subject) if settings['cache_root'] else None
            if (directory/'manifest.json').exists() or cache is None or not (cache/'manifest.json').exists():
                args.append('--resume')
            else:
                args += ['--resume-from', str(cache)]
            with log.open('a', buffering=1) as file, contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
                mesh_fit.main(args, on_scan_done=lambda record: events.put(dict(event='scan', subject=subject, record=record)))
            manifest = json.loads((directory/'manifest.json').read_text())
            events.put(dict(event='finished', subject=subject, records=manifest['scans']))
        except (Exception, SystemExit):
            error = traceback.format_exc()
            with log.open('a') as file:
                file.write(error)
            events.put(dict(event='failed', subject=subject, error=error))
        finally:
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--scan-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--devices', required=True, help='comma-separated physical CUDA indices')
    parser.add_argument('--workers-per-device', type=int, default=1,
                        help='independent patient workers per listed GPU')
    parser.add_argument('--subjects-file', type=Path, help='engineering QC examples only')
    parser.add_argument('--cache-root', type=Path, help='reuse fully verified pilot prefixes')
    parser.add_argument('--scope', choices=['engineering-qc', 'cohort-candidates'], required=True)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261006)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) and not output.is_relative_to(Path('/data5/take/brainode')):
        parser.error('output must remain under the project or its data5 directory')
    if args.scope == 'cohort-candidates' and args.subjects_file:
        parser.error('full baseline mode must retain all subjects in the table')
    if args.cache_root and (args.cache_root/'progress.json').exists():
        cached_status = json.loads((args.cache_root/'progress.json').read_text())['status']
        if cached_status == 'running':
            parser.error('cache-root still has running workers; finish that batch first')
    if args.iterations < 1 or args.workers_per_device < 1:
        parser.error('iterations and workers-per-device must be positive')
    devices = [int(x) for x in args.devices.split(',')]
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error('distinct nonnegative GPU indices required')
    with args.scan_table.open() as file:
        rows = list(csv.DictReader(file))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['pid']].append(row)
    grouped = {pid: ordered_scans(group) for pid, group in grouped.items()}
    subjects = set(grouped)
    if args.subjects_file:
        subjects = set(args.subjects_file.read_text().splitlines())
        if not subjects or not subjects <= set(grouped):
            parser.error('QC subject IDs must occur in the baseline table')
    table = freeze_sources(args.snapshot.resolve(), args.scan_table.resolve())
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output/'logs').mkdir(exist_ok=True)
    plan = dict(scope=args.scope, baseline_subjects=len(grouped), baseline_scans=len(rows),
                scheduled_subjects=len(subjects), scheduled_scans=sum(len(grouped[p]) for p in subjects),
                subjects=sorted(subjects), devices=devices, workers_per_device=args.workers_per_device,
                iterations=args.iterations, seed=args.seed,
                table=str(table), table_sha256=digest(table), code_snapshot=str(args.snapshot.resolve()),
                ordering=ORDERING_POLICY, initialization='first template mask-volume; then previous fitted observation',
                same_day_repeats='all retained; no time adjustment', qc_thresholds=QC_THRESHOLDS,
                qc_flags_are_exclusion_rules=False, scientific_training_adoption=False,
                cache_root=str(args.cache_root.resolve()) if args.cache_root else None)
    if (output/'plan.json').exists():
        old = json.loads((output/'plan.json').read_text())
        for key in ('scope','subjects','iterations','seed','table_sha256','code_snapshot','ordering'):
            if old[key] != plan[key]:
                raise ValueError(f'existing batch plan differs: {key}')
    write_json(output/'plan.json', plan)
    context = mp.get_context('spawn')
    tasks, events = context.Queue(), context.Queue()
    for subject in sorted(subjects, key=lambda p: (-len(grouped[p]), p)):
        tasks.put((subject, str(output/subject_folder(subject))))
    worker_devices = [device for device in devices for _ in range(args.workers_per_device)]
    for _ in worker_devices:
        tasks.put(None)
    settings = dict(snapshot=str(args.snapshot.resolve()), table=str(table), output=str(output),
                    scope=args.scope, iterations=args.iterations, seed=args.seed,
                    cache_root=plan['cache_root'])
    processes = [context.Process(target=worker, args=(device,tasks,events,settings)) for device in worker_devices]
    state = dict(status='running', controller_pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 workers={}, subjects={subject:'queued' for subject in sorted(subjects)},
                 completed_subjects=0, failed_subjects=0, completed_scans=0, flagged_scans=0,
                 scheduled_scans=plan['scheduled_scans'], scheduled_subjects=len(subjects), errors={})
    records = {}
    started = time.monotonic()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for slot, (device, process) in enumerate(zip(worker_devices,processes)):
            process.start()
            state['workers'][str(slot)] = dict(device=device,pid=process.pid)
        while state['completed_subjects']+state['failed_subjects'] < len(subjects):
            try:
                event = events.get(timeout=10)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    raise RuntimeError('all workers exited before completing the plan')
                event = None
            if event:
                kind = event['event']
                if kind == 'fatal_worker':
                    raise RuntimeError(event['error'])
                subject = event['subject']
                if kind == 'started':
                    state['subjects'][subject] = f'running on GPU {event["device"]}'
                elif kind in ('scan', 'finished'):
                    received = [event['record']] if kind == 'scan' else event['records']
                    for record in received:
                        records[(subject,record['uid'])] = record
                    if kind == 'finished':
                        state['subjects'][subject] = 'completed_candidates'
                        state['completed_subjects'] += 1
                elif kind == 'failed':
                    state['subjects'][subject] = 'failed; observations retained in plan'
                    state['failed_subjects'] += 1
                    state['errors'][subject] = event['error']
            state.update(completed_scans=len(records), flagged_scans=sum(bool(r['qc_flags']) for r in records.values()),
                         elapsed_seconds=time.monotonic()-started,
                         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write_json(output/'progress.json', state)
        state['status'] = 'completed_candidates' if not state['failed_subjects'] else 'completed_with_failures'
    except KeyboardInterrupt:
        state['status'] = 'interrupted; verified prefixes retained'
    except Exception:
        state.update(status='controller_failed',controller_error=traceback.format_exc())
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        write_json(output/'progress.json', state)
        write_json(output/'qc_records.json', [dict(subject=p,**record) for (p,u),record in sorted(records.items())])
        if records:
            flags = Counter(flag for record in records.values() for flag in record['qc_flags'])
            write_json(output/'qc_summary.json', dict(completed_scans=len(records), flagged_scans=state['flagged_scans'],
                                                      flag_counts=dict(flags), training_adoption=False))
    print(json.dumps({k:state[k] for k in ('status','completed_subjects','failed_subjects','completed_scans','flagged_scans')},indent=2))


if __name__ == '__main__':
    main()
