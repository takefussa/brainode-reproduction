"""Measure short extra fitting workers alongside a running mesh batch.

Diagnostic optimizations never produce adopted mesh candidates. The observed
production CSVs and input NPZ remain read-only.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]


def observations(root, device):
    sys.path.insert(0, str(PROJECT))
    from mesh_cohort import subject_folder
    progress = json.loads((root/'progress.json').read_text())
    result = {}
    for subject, state in progress['subjects'].items():
        if state != f'running on GPU {device}':
            continue
        directory = root/subject_folder(subject)
        paths = list(directory.glob('*_loss.csv'))
        if not paths:
            continue
        path = max(paths, key=lambda p: p.stat().st_mtime)
        with path.open() as file:
            rows = list(csv.DictReader(file))
        rows = [row for row in rows if row.get('seconds') and row.get('iteration')]
        if not rows:
            continue
        latest = rows[-1]
        earlier = rows[max(0, len(rows)-501)]
        seconds = float(latest['seconds'])-float(earlier['seconds'])
        result[subject] = dict(path=str(path), iteration=int(latest['iteration']),
            seconds=float(latest['seconds']),
            recent_updates_per_second=(int(latest['iteration'])-int(earlier['iteration']))/seconds if seconds else None)
    return result


def probe(snapshot, data_path, output, device, iterations, index):
    os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import torch
    from mesh_fit import reconstruct_scan
    torch.set_num_threads(4)
    with np.load(data_path) as data:
        source, faces, target = (data[key].copy() for key in ('source_vertices','faces','target'))
    _, elapsed = reconstruct_scan(source, faces, target, iterations, f'cuda:{device}',
        20261006+index, output/f'probe_{index}_loss.csv', 'fixed-source', 'public')
    (output/f'probe_{index}_result.json').write_text(json.dumps(dict(iterations=iterations,
        elapsed_seconds=elapsed, updates_per_second=iterations/elapsed, device=device)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--extra-workers', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('diagnostic reports must remain under the home project')
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.run_root/'plan.json').read_text())
    if args.device not in plan['devices'] or args.extra_workers < 1 or args.iterations < 1:
        parser.error('benchmark requires a production GPU and positive worker/update counts')
    before = observations(args.run_root, args.device)
    context = mp.get_context('spawn')
    processes = [context.Process(target=probe, args=(Path(plan['code_snapshot']), args.input,
        output, args.device, args.iterations, index)) for index in range(args.extra_workers)]
    started = time.monotonic()
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError('benchmark probe failed')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    elapsed = time.monotonic()-started
    after = observations(args.run_root, args.device)
    probes = [json.loads((output/f'probe_{index}_result.json').read_text()) for index in range(args.extra_workers)]
    updates = {}
    for subject in before.keys() & after.keys():
        a,b = before[subject],after[subject]
        if a['path'] == b['path']:
            updates[subject] = b['iteration']-a['iteration']
    baseline = sum(r['recent_updates_per_second'] or 0 for r in before.values())
    aggregate = (sum(updates.values())+args.iterations*len(probes))/elapsed
    result = dict(production_devices=plan['devices'], benchmark_device=args.device,
        baseline_workers_on_device=plan['workers_per_device'], extra_workers=len(probes),
        scope='short diagnostic fitting only; no adopted meshes', input=str(args.input),
        iterations_per_probe=args.iterations, elapsed_seconds_including_startup=elapsed,
        before=before, after=after, observed_production_updates=updates, probes=probes,
        baseline_recent_updates_per_second=baseline,
        combined_updates_per_second_including_startup=aggregate,
        estimated_throughput_ratio=aggregate/baseline if baseline else None,
        caveat='single short concurrent measurement; different production target shapes; startup included')
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
