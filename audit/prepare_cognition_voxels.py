"""Prepare every ADNI1 PCA observation on a fixed voxel-center lattice."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import trimesh

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from cognition import PCAShapeVoxelizer
from lab_shapes import ShapePCA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    root,output,report=args.data_root.resolve(),args.output.resolve(),args.report.resolve()
    if not output.is_relative_to(Path('/data5/take/brainode')) or not report.is_relative_to(PROJECT):
        parser.error('voxel data belong under data5 and light reports under the home project')
    output.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with np.load(root/'observations.npz',allow_pickle=False) as file:
        data={k:file[k].copy() for k in file.files}
    if len(np.unique(data['uids']))!=len(data['uids']):
        raise ValueError('nonunique observation UIDs')
    pca=ShapePCA.load(root/'pca.npz')
    train=data['splits']=='train'
    vertices=pca.inverse_transform(data['coefficients'][train]).reshape(-1,1490,3)
    lower=np.minimum((vertices*.98).min((0,1)),(vertices*1.02).min((0,1)))
    upper=np.maximum((vertices*.98).max((0,1)),(vertices*1.02).max((0,1)))
    origin=np.floor((lower+upper)/2-31.5)
    config=dict(data_root=str(root),method='center-occupancy',pitch_mm=.5,grid_shape=[128,128,128],
        origin_mm=origin.tolist(),grid_fit='train augmented PCA observations only',
        train_augmented_bounds=[lower.tolist(),upper.tolist()],occupancy_rays=[3,7],workers=args.workers,
        observations=len(data['uids']),source_files={p.name:digest(p) for p in root.glob('*.npz')},
        code_sha256={str(p.relative_to(PROJECT)):digest(p) for p in (PROJECT/'cognition.py',PROJECT/'lab_shapes.py',Path(__file__))},
        open3d_version=o3d.__version__,spatial_unit_assumption='one numerical coordinate unit interpreted as one mm',
        selection='all 1405 CN/AD observations, same patient split and all repeats retained',cnn_training_started=False)
    atomic_json(report/'config.json',config);atomic_json(output/'config.json',config)
    records=[];failures=[]
    def build(index):
        coefficients=data['coefficients'][index]
        first=PCAShapeVoxelizer(pca,data['faces'],grid_shape=(128,128,128),origin_mm=origin,pitch_mm=.5,
                                method='center-occupancy',occupancy_rays=3)(coefficients)[0,0].astype(np.uint8)
        second=PCAShapeVoxelizer(pca,data['faces'],grid_shape=(128,128,128),origin_mm=origin,pitch_mm=.5,
                                 method='center-occupancy',occupancy_rays=7)(coefficients)[0,0].astype(np.uint8)
        v=pca.inverse_transform(coefficients[None]).reshape(-1,3)
        volume=float(trimesh.Trimesh(vertices=v,faces=data['faces'],process=False).volume)
        uid=str(data['uids'][index]);path=output/f'{uid}.npz'
        temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as file:
            np.savez_compressed(file,occupancy=first)
        temporary.replace(path)
        return dict(index=index,pid=str(data['pids'][index]),uid=uid,split=str(data['splits'][index]),
            observed_c=float(data['conditions'][index]),time=float(data['times'][index]),
            original_quality_flags=str(data['qc_flags'][index]),occupied_voxels=int(first.sum()),
            pca_mesh_volume_mm3=volume,voxel_volume_bias_percent=float(100*(first.sum()*.125/volume-1)),
            ray_vote_disagreement_voxels=int(np.count_nonzero(first!=second)),
            file=str(path),sha256=digest(path))
    def update(status):
        progress=dict(status=status,planned_scans=len(data['uids']),completed_scans=len(records),
            failed_scans=len(failures),failures=failures,observations_excluded=0,
            elapsed_seconds=time.monotonic()-started,cnn_training_started=False)
        if records:
            progress.update(max_abs_voxel_volume_bias_percent=max(abs(r['voxel_volume_bias_percent']) for r in records),
                scans_with_ray_vote_disagreement=sum(r['ray_vote_disagreement_voxels']>0 for r in records))
        atomic_json(report/'progress.json',progress)
        print(json.dumps(progress),flush=True)
    update('preparing')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(build,i):i for i in range(len(data['uids']))}
        for future in as_completed(futures):
            index=futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                failures.append(dict(index=index,pid=str(data['pids'][index]),uid=str(data['uids'][index]),error=repr(error)))
            if (len(records)+len(failures))%50==0:
                update('preparing')
    records.sort(key=lambda r:r['index'])
    atomic_json(report/'manifest.json',dict(config=config,records=records,failures=failures))
    atomic_json(output/'manifest.json',dict(config=config,records=records,failures=failures))
    update('completed_voxel_candidates' if not failures else 'voxel_preparation_incomplete')
    if failures:
        raise RuntimeError('some observations could not be voxelized; no automatic exclusion or crop applied')


if __name__=='__main__':
    main()
