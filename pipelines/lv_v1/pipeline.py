"""Unattended, restartable bilateral LV reproduction. Original data are read-only.

Each stage has an atomic completion marker with output hashes. There is no
test-based checkpoint selection, failed-subject exclusion, or automatic repair.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SIDES = ('left', 'right')
LABELS = {'left': [163,164,165,166,167], 'right': [168,169,170,171,172]}
TRAIN_FILES = ['train_lab.py','model.py','paper_training.py','evaluation.py',
               'lab_shapes.py','resume_utils.py']
CONT_FILES = TRAIN_FILES + ['train_continuous.py','paper_cognition.py','cognition.py','deterministic_pool.py','cnn_microbatch.py']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for part in iter(lambda: file.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(newline='') as file: return list(csv.DictReader(file))


def paths(c, side):
    return dict(mesh=Path(c['heavy'])/f'mesh_{side}', data=Path(c['heavy'])/f'pca_{side}',
                voxels=Path(c['heavy'])/f'voxels_{side}', initial=Path(c['heavy'])/f'observed_{side}',
                continuous=Path(c['heavy'])/f'continuous_{side}',
                report=Path(c['report'])/side)


def ordered(c):
    from mesh_cohort import ordered_scans
    grouped = defaultdict(list)
    for row in read_rows(c['table']): grouped[row['pid']].append(row)
    return {pid: ordered_scans(group) for pid, group in sorted(grouped.items())}


def preflight(c):
    if len(c['gpus']) != 2 or len(set(c['gpus'])) != 2: raise ValueError('exactly two GPU devices required')
    for relative, expected in c['snapshot_sha256'].items():
        if digest(HERE/relative) != expected: raise ValueError('frozen code/input changed: '+relative)
    for path, expected in c.get('runtime_sha256', {}).items():
        if digest(path) != expected: raise ValueError('frozen CUDA runtime changed: '+path)
    rows = read_rows(c['table']); groups = ordered(c)
    if len(rows) != c['scans'] or len(groups) != c['patients']: raise ValueError('cohort coverage changed')
    if Counter(r['split'] for r in rows) != c['split_scans']: raise ValueError('split counts changed')
    if any(r['class'] not in ('CN','AD') for r in rows): raise ValueError('unexpected diagnosis')
    # Snapshot table is identical to the reviewed baseline; verify every mask
    # at worker time. Full raw MRI is not used to build cavity masks.
    if digest(c['table']) != c['table_sha256']: raise ValueError('baseline table changed')
    if not Path(c['heavy']).is_relative_to('/data5/take/brainode'): raise ValueError('heavy storage root')


def mesh_worker(gpu, tasks, events, c):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    from mesh_cohort import subject_folder
    import lv_mesh_fit
    import torch
    while True:
        task = tasks.get()
        if task is None: return
        side, pid = task; p = paths(c, side)
        log = p['mesh']/'logs'/f'{subject_folder(pid)}.log'
        events.put(dict(event='started', side=side, pid=pid, gpu=gpu, process=os.getpid()))
        import contextlib
        try:
            args = ['--subject',pid,'--side',side,'--scan-table',c['table'],
                    '--template',str(HERE/'assets'/f'lv_{side}_template.npz'),
                    '--template-audit',str(HERE/'assets/template_audit.json'),
                    '--output',str(p['mesh']/subject_folder(pid)), '--device','cuda:0',
                    '--iterations',str(c['mesh_iterations']), '--seed',str(c['mesh_seed']),
                    '--scope','cohort-candidates','--resume']
            with log.open('a', buffering=1) as file, contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
                lv_mesh_fit.main(args, on_scan_done=lambda r: events.put(dict(event='scan', side=side, pid=pid, uid=r['uid'])))
            events.put(dict(event='finished', side=side, pid=pid))
        except BaseException:
            error = traceback.format_exc()
            with log.open('a') as file: file.write(error)
            events.put(dict(event='failed', side=side, pid=pid, error=error))
        finally: torch.cuda.empty_cache()


def fit_meshes(c):
    from mesh_cohort import subject_folder
    groups = ordered(c); report = Path(c['report'])
    ctx = mp.get_context('spawn'); events = ctx.Queue(); tasks = ctx.Queue()
    jobs = [(side,pid) for pid in sorted(groups, key=lambda pid:(-len(groups[pid]),pid)) for side in SIDES]
    for side in SIDES:
        (paths(c,side)['mesh']/'logs').mkdir(parents=True,exist_ok=True)
    for job in jobs: tasks.put(job)
    worker_gpus = [gpu for gpu in c['gpus'] for _ in range(c['mesh_workers_per_gpu'])]
    for _ in worker_gpus: tasks.put(None)
    workers = [ctx.Process(target=mesh_worker,args=(gpu,tasks,events,c)) for gpu in worker_gpus]
    done = set(); scans = set(); active = {}; started = time.monotonic()
    # Count persisted observations for accurate restart progress, while each
    # worker still verifies its complete cached chronological prefix.
    for side,pid in jobs:
        m = paths(c,side)['mesh']/subject_folder(pid)/'manifest.json'
        if m.exists():
            for r in json.loads(m.read_text())['scans']: scans.add((side,pid,r['uid']))
    original_count = len(scans)
    try:
        for worker in workers: worker.start()
        while len(done) < len(jobs):
            try: e = events.get(timeout=15)
            except queue.Empty:
                e = None
                if not any(w.is_alive() for w in workers): raise RuntimeError('all mesh workers exited before coverage completion')
            if e:
                key = (e['side'],e['pid']); kind=e['event']
                if kind=='failed': raise RuntimeError('mesh failure; no exclusion: '+json.dumps(e))
                if kind=='started': active[str(e['process'])]=e
                if kind=='scan': scans.add((*key,e['uid']))
                if kind=='finished':
                    done.add(key)
                    m=paths(c,e['side'])['mesh']/subject_folder(e['pid'])/'manifest.json'
                    records=json.loads(m.read_text())['scans']
                    if len(records)!=len(groups[e['pid']]): raise ValueError('subject coverage mismatch')
                    scans.update((*key,r['uid']) for r in records)
                    active={k:v for k,v in active.items() if (v['side'],v['pid'])!=key}
            elapsed=time.monotonic()-started; newly=len(scans)-original_count
            state=dict(status='running', completed_meshes=len(scans),total_meshes=len(jobs) and 2*c['scans'],
                completed_patient_sides=len(done), total_patient_sides=len(jobs),
                side_counts={side:sum(s[0]==side for s in scans) for side in SIDES},
                active_workers=active, worker_pids=[w.pid for w in workers], physical_gpus=c['gpus'],
                elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed/newly*(2*c['scans']-len(scans)) if newly else None,
                updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(report/'mesh_progress.json',state)
        if len(scans)!=2*c['scans']: raise ValueError('full bilateral scan coverage not achieved')
        state['status']='completed'; write_json(report/'mesh_progress.json',state)
    finally:
        for worker in workers:
            if worker.is_alive(): worker.terminate()
            worker.join(timeout=10)
    return [paths(c,side)['mesh']/subject_folder(pid)/'manifest.json' for side,pid in jobs]


def prepare_data(c,side):
    import numpy as np
    import open3d as o3d
    import trimesh
    from lab_shapes import fit_train_pca, patient_indices
    from mesh_cohort import subject_folder
    p=paths(c,side); p['report'].mkdir(parents=True,exist_ok=True)
    archive_partial(p['data'])
    p['data'].mkdir(parents=True)
    records=[]; vertex_arrays=[]; manifests={}; faces=None
    for pid,rows in ordered(c).items():
        directory=p['mesh']/subject_folder(pid); manifest_path=directory/'manifest.json'
        manifest=json.loads(manifest_path.read_text()); manifests[str(manifest_path)]=digest(manifest_path)
        if manifest['labels']!=LABELS[side] or manifest['iterations']!=c['mesh_iterations']: raise ValueError('mesh configuration mismatch')
        if [r['uid'] for r in manifest['scans']]!=[r['uid'] for r in rows]: raise ValueError('mesh order/coverage mismatch')
        previous=None
        for row,r in zip(rows,manifest['scans']):
            for name,expected in r['artifact_sha256'].items():
                if digest(directory/name)!=expected: raise ValueError('mesh artifact changed: '+name)
            if digest(row['parcellation_path'])!=r['source_sha256']: raise ValueError('source segmentation changed')
            with np.load(directory/(r['uid']+'.npz'),allow_pickle=False) as a:
                v=a['vertices'].copy(); f=a['faces'].copy()
                if previous is not None: np.testing.assert_array_equal(a['source_vertices'],previous)
            if faces is None: faces=f
            np.testing.assert_array_equal(f,faces)
            if v.shape!=(2490,3) or not np.isfinite(v).all(): raise ValueError('invalid corresponding LV shape')
            mesh=trimesh.Trimesh(v,f,process=False)
            if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0: raise ValueError('invalid observed LV geometry; no silent exclusion')
            previous=v; vertex_arrays.append(v)
            records.append(dict(**row,qc_flags=';'.join(r['qc_flags']),mesh_npz=str(directory/(r['uid']+'.npz')),
                                mesh_sha256=r['artifact_sha256'][r['uid']+'.npz'],dice=r['overlap']['dice'],
                                volume_bias_percent=r['volume_bias_percent'],self_intersection_pairs=r['after']['self_intersecting_triangle_pairs']))
    vertices=np.stack(vertex_arrays).astype(np.float32)
    pids=np.array([r['pid'] for r in records]); splits=np.array([r['split'] for r in records])
    if len(records)!=c['scans']: raise ValueError('PCA input coverage')
    print('PCA full SVD:',side,vertices.shape,flush=True)
    pca,coefficients,fitted=fit_train_pca(vertices,pids,splits,150)
    reconstructed=pca.inverse_transform(coefficients).reshape(vertices.shape)
    errors=np.linalg.norm(reconstructed-vertices,axis=2).mean(1); crosses=[]
    for index,v in enumerate(reconstructed):
        m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(faces))
        crosses.append(len(m.get_self_intersecting_triangles()))
        records[index].update(pca_mean_vertex_error_mm=float(errors[index]),pca_self_intersection_pairs=crosses[-1])
    with (p['data']/'pca.npz').open('wb') as file:
        np.savez_compressed(file,mean=pca.mean_,components=pca.components_,explained_variance=fitted.explained_variance_,
                            explained_variance_ratio=fitted.explained_variance_ratio_,singular_values=fitted.singular_values_)
    np.savez_compressed(p['data']/'observations.npz',vertices=vertices,coefficients=coefficients,faces=faces,
        times=np.array([float(r['time']) for r in records]),conditions=np.array([float(r['observed_c']) for r in records]),
        pids=pids,splits=splits,uids=np.array([r['uid'] for r in records]),days=np.array([r['day'] for r in records]),
        ages=np.array([float(r['age']) for r in records]),qc_flags=np.array([r['qc_flags'] for r in records]))
    with (p['report']/'observations.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=dict(region=side,cohort='ADNI1',observations=len(records),dimensions=150,vertex_count=vertices.shape[1],
        face_count=len(faces),train_only_pca=True,pca_fit_scans=int(sum(splits=='train')),
        explained_variance_ratio_sum=float(fitted.explained_variance_ratio_.sum()),
        pca_surface_intersections=sum(x>0 for x in crosses),
        reconstruction_mean_vertex_error_mm={s:float(errors[splits==s].mean()) for s in ('train','test')},
        mesh_dice={s:float(np.mean([r['dice'] for r in records if r['split']==s])) for s in ('train','test')},
        flagged_observations_retained=sum(bool(r['qc_flags']) for r in records),observations_excluded=0,
        source_manifests_sha256=manifests,table_sha256=c['table_sha256'],spatial_unit_assumption='1 numeric unit = 1 mm')
    write_json(p['data']/'summary.json',summary);write_json(p['report']/'data_summary.json',summary)
    # A single physical grid per side, fitted only to all training PCA shapes.
    # Pad each axis by 25% of its extent plus 40 mm total, round pitch upward
    # to 1/8 mm and origin to the pitch lattice. No test fitting or clipping.
    train=reconstructed[splits=='train']
    a=np.minimum((train*.98).min((0,1)),(train*1.02).min((0,1)))
    b=np.maximum((train*.98).max((0,1)),(train*1.02).max((0,1)))
    extent=b-a; required=extent*1.25+40
    pitch=float(np.ceil(required.max()/127/.125)*.125)
    center=(a+b)/2; origin=np.floor((center-63.5*pitch)/pitch)*pitch
    grid=dict(grid_shape=[128]*3,origin_mm=origin.tolist(),pitch_mm=pitch,method='center-occupancy')
    write_json(p['report']/'grid.json',dict(grid=grid,train_bounds_mm=[a.tolist(),b.tolist()],
        padding_policy='25% extent plus 40 mm total; isotropic pitch ceil to .125 mm; train only',
        test_grid_refit=False,per_patient_rescaling=False,cropping=False))
    return [p['data']/'observations.npz',p['data']/'pca.npz',p['data']/'summary.json',p['report']/'observations.csv',p['report']/'grid.json']


def voxel_cache(c,side):
    import numpy as np
    import trimesh
    from concurrent.futures import ThreadPoolExecutor
    from cognition import PCAShapeVoxelizer
    from lab_shapes import ShapePCA
    p=paths(c,side); archive_partial(p['voxels']);p['voxels'].mkdir(parents=True)
    with np.load(p['data']/'observations.npz',allow_pickle=False) as file: data={k:file[k] for k in file.files}
    pca=ShapePCA.load(p['data']/'pca.npz');grid=json.loads((p['report']/'grid.json').read_text())['grid']
    voxel3=PCAShapeVoxelizer(pca,data['faces'],**grid,occupancy_rays=3)
    voxel7=PCAShapeVoxelizer(pca,data['faces'],**grid,occupancy_rays=7)
    def one(i):
        a=voxel3(data['coefficients'][i]);b=voxel7(data['coefficients'][i])
        disagreement=int(np.count_nonzero(a!=b))
        if disagreement: raise ValueError(f'3/7 ray disagreement {side} UID={data["uids"][i]}: {disagreement}; no silent exclusion')
        file=p['voxels']/f'{i:04d}_{data["uids"][i]}.npz'
        np.savez_compressed(file,occupancy=a[0,0].astype(np.uint8))
        vertices=pca.inverse_transform(data['coefficients'][i:i+1]).reshape(-1,3)
        volume=float(trimesh.Trimesh(vertices,data['faces'],process=False).volume)
        return dict(index=i,pid=str(data['pids'][i]),uid=str(data['uids'][i]),split=str(data['splits'][i]),
            observed_c=float(data['conditions'][i]),time=float(data['times'][i]),original_quality_flags=str(data['qc_flags'][i]),
            occupied_voxels=int(a.sum()),voxel_volume_bias_percent=float(100*(a.sum()*grid['pitch_mm']**3/volume-1)),
            ray_vote_disagreement_voxels=disagreement,file=str(file),sha256=digest(file))
    records=[]
    with ThreadPoolExecutor(max_workers=c['voxel_workers']) as pool:
        for record in pool.map(one,range(len(data['pids']))):
            records.append(record)
            if len(records)%25==0:
                write_json(p['report']/'voxel_progress.json',dict(status='running',completed=len(records),total=len(data['pids'])))
    manifest=dict(region=side,grid=grid,records=records,scans=len(records),grid_fit_train_only=True,
        observations_sha256=digest(p['data']/'observations.npz'),pca_sha256=digest(p['data']/'pca.npz'),
        excluded=0,ray_vote_disagreements=0)
    write_json(p['voxels']/'manifest.json',manifest)
    write_json(p['report']/'voxel_progress.json',dict(status='completed',completed=len(records),total=len(records)))
    return [p['voxels']/'manifest.json']+[Path(r['file']) for r in records]


def config_file(c,side,continuous=False):
    p=paths(c,side)
    source=HERE/'assets'/('continuous_recipe.json' if continuous else 'observed_recipe.json')
    recipe=json.loads(source.read_text());recipe.update(experiment=f'adni1_lv_{side}_v1',data_root=str(p['data']),
        output=str(p['continuous'] if continuous else p['initial']),report=str(p['report']/('continuous' if continuous else 'observed')),
        code_sha256={name:digest(HERE/name) for name in (CONT_FILES if continuous else TRAIN_FILES)},
        data_sha256={name:digest(p['data']/name) for name in ('observations.npz','pca.npz','summary.json')},region=side)
    gpu=c['gpus'][SIDES.index(side)]
    if continuous:
        recipe.update(initial_checkpoint=str(p['initial']/'latest.pt'),initial_checkpoint_sha256=digest(p['initial']/'latest.pt'),
                      voxel_manifest=str(p['voxels']/'manifest.json'),voxel_manifest_sha256=digest(p['voxels']/'manifest.json'),
                      grid=json.loads((p['report']/'grid.json').read_text())['grid'])
        recipe['cnn']['physical_gpu']=gpu;recipe['ode']['physical_gpu']=gpu
    else:
        recipe['gpu_physical_index']=gpu;recipe['other_reserved_gpu']=c['gpus'][1-SIDES.index(side)]
        summary=json.loads((p['data']/'summary.json').read_text());recipe['pca_surface_intersections']=summary['pca_surface_intersections']
    path=p['report']/('continuous_config.json' if continuous else 'observed_config.json')
    if path.exists() and json.loads(path.read_text())!=recipe: raise ValueError('existing stage config differs')
    write_json(path,recipe)
    return path


def train_stage(c,side,stage):
    p=paths(c,side); continuous=stage!='observed'; config=config_file(c,side,continuous)
    output=p['continuous']/stage if continuous else p['initial']
    if stage in ('pseudo','evaluate'): archive_partial(output)
    script='train_continuous.py' if continuous else 'train_lab.py'
    args=[sys.executable,str(HERE/script)]+([stage] if continuous else [])+['--config',str(config)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(c['gpus'][SIDES.index(side)]),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    log=p['report']/f'{stage}.log'
    with log.open('a',buffering=1) as file:
        result=subprocess.run(args,cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{side} {stage} failed ({result.returncode}); see {log}')
    report=p['report']/('continuous/'+stage if continuous else 'observed')
    status=json.loads((report/'progress.json').read_text())
    if not status['status'].startswith('completed'): raise ValueError('stage did not reach completion')
    files=[report/'progress.json',config]
    if stage in ('observed','cnn','ode'): files.append(output/'latest.pt')
    if stage in ('observed','evaluate'): files.extend([report/'evaluation.json',report/'future_metrics.csv',output/'future_predictions.npz'])
    if stage=='cnn': files.extend([report/'evaluation.json',report/'cognition_predictions.csv'])
    if stage=='pseudo':
        files.extend([output/'manifest.json',output/'samples.npz'])
        files.extend(Path(r['file']) for r in json.loads((output/'manifest.json').read_text())['records'])
    return files


def archive_partial(directory):
    """Preserve an unfinished noncheckpoint stage; never touch source data."""
    directory=Path(directory)
    if directory.exists():
        if not directory.is_relative_to('/data5/take/brainode'): raise ValueError('unsafe archive path')
        directory.rename(directory.with_name(directory.name+'.incomplete.'+str(time.time_ns())))


def stage_worker(c,side,stage):
    functions={'data':prepare_data,'voxels':voxel_cache}
    function=functions.get(stage)
    return function(c,side) if function else train_stage(c,side,stage)


def run_one(c,key,function):
    lock_path=Path(c['report'])/'stage_locks'/f'{key}.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return run_one_locked(c,key,function)


def run_one_locked(c,key,function):
    report=Path(c['report']);marker=report/'completed'/f'{key}.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        for path,expected in saved['files_sha256'].items():
            if digest(path)!=expected: raise ValueError('completed stage output changed: '+path)
        return
    write_json(report/'stage_status'/f'{key}.json',dict(status='running',started_utc=datetime.now(timezone.utc).isoformat()))
    for attempt in range(1,4):
        try:
            files=function()
            write_json(marker,dict(status='completed',files_sha256={str(p):digest(p) for p in files},
                                   completed_utc=datetime.now(timezone.utc).isoformat()))
            write_json(report/'stage_status'/f'{key}.json',dict(status='completed',attempt=attempt))
            return
        except Exception:
            error=traceback.format_exc()
            write_json(report/'stage_status'/f'{key}.json',dict(status='retrying' if attempt<3 else 'failed',attempt=attempt,error=error,
                      updated_utc=datetime.now(timezone.utc).isoformat()))
            print(key,error,flush=True)
            if attempt==3: raise
            time.sleep(10)


def side_chain(c,side):
    for stage in ('data','observed','voxels','pseudo','cnn','ode','evaluate'):
        run_one(c,side+'_'+stage,lambda:stage_worker(c,side,stage))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();c=json.loads(args.config.read_text());report=Path(c['report']);report.mkdir(parents=True,exist_ok=True)
    lock=(report/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='preflight',pid=os.getpid(),config_sha256=digest(args.config),physical_gpus=c['gpus'],
               started_utc=datetime.now(timezone.utc).isoformat(),code_snapshot=str(HERE))
    def update(**fields):
        state.update(fields,updated_utc=datetime.now(timezone.utc).isoformat());write_json(report/'progress.json',state)
    update()
    try:
        preflight(c);update(status='running',stage='bilateral_mesh_fitting')
        run_one(c,'meshes',lambda:fit_meshes(c))
        update(stage='separate_side_training_and_forecasting')
        # Separate processes avoid changing CUDA visibility after CUDA import.
        ctx=mp.get_context('spawn');workers=[ctx.Process(target=side_chain,args=(c,side)) for side in SIDES]
        for worker in workers: worker.start()
        while any(w.is_alive() for w in workers):
            update(side_processes={side:dict(pid=w.pid,alive=w.is_alive(),exitcode=w.exitcode) for side,w in zip(SIDES,workers)})
            time.sleep(15)
        for worker in workers: worker.join()
        if any(w.exitcode!=0 for w in workers): raise RuntimeError('side chain failed; see stage_status and side logs; no exclusions')
        update(stage='score_comparison_and_3d_export')
        from final_export import export_all
        run_one(c,'review_bundle',lambda:export_all(c))
        update(status='completed',stage='completed',future_mesh_bundle=c['bundle_zip'])
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__': main()
