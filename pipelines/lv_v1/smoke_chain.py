"""Functional test only: synthetic LV coefficients, not research results.

Exercises true LV vertex count, physical pitch != .5, all training stages,
epoch restore, Algorithm 2, and portable export in isolated test directories.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from pipeline import HERE,SIDES,paths,write_json,digest,voxel_cache,TRAIN_FILES,CONT_FILES,train_stage
from lab_shapes import ShapePCA


def setup(root,report):
    c=dict(heavy=str(root),report=str(report),gpus=[0,6],voxel_workers=2,test_future_targets=2,FUNCTIONAL_TEST_ONLY=True,
           bundle=str(report/'FUNCTIONAL_TEST_ONLY_bundle'),bundle_zip=str(report/'FUNCTIONAL_TEST_ONLY_bundle.zip'),
           table=str(HERE/'assets/eligible.csv'),review_source_uid='34537',review_target_uid='176917',
           scans=8,patients=4,mesh_iterations=5000)
    from mesh_cohort import subject_folder
    for side in SIDES:
        p=paths(c,side);p['report'].mkdir(parents=True);p['data'].mkdir(parents=True)
        with np.load(HERE/'assets'/f'lv_{side}_template.npz') as a: vertices=a['vertices'];faces=a['faces']
        flat=vertices.ravel(); rng=np.random.default_rng(7)
        matrix=rng.normal(size=(len(flat),150));matrix[:,0]=flat
        q,_=np.linalg.qr(matrix);components=q.T
        coeff=np.zeros((8,150));coeff[:,0]=[0,3,0,2,0,4,0,3]
        coeff[:,1]=[0,.01,0,.02,0,.01,0,.02]
        reconstructed=(coeff@components+flat).reshape(8,-1,3).astype(np.float32)
        pids=np.repeat(['test:train_convert','test:train_CN','ADNI:RID:352','test:test_AD'],2)
        splits=np.repeat(['train','train','test','test'],2)
        uids=np.array(['t1','t2','t3','t4','34537','176917','t7','t8'])
        np.savez_compressed(p['data']/'pca.npz',mean=flat,components=components)
        np.savez_compressed(p['data']/'observations.npz',vertices=reconstructed,coefficients=coeff,faces=faces,
            pids=pids,splits=splits,uids=uids,days=np.array(['2006-04-07','2010-04-28']*4),
            times=np.tile([.4,.5],4),ages=np.tile([70,74.06],4),conditions=np.array([0,1,0,0,0,0,1,1]),qc_flags=np.repeat('',8))
        write_json(p['data']/'summary.json',dict(FUNCTIONAL_TEST_ONLY=True,pca_surface_intersections=0))
        grid=dict(grid_shape=[128]*3,origin_mm=[-96]*3,pitch_mm=1.5,method='center-occupancy')
        write_json(p['report']/'grid.json',dict(grid=grid,FUNCTIONAL_TEST_ONLY=True))
        voxel_cache(c,side)
        # Export test requires source/future transformations. Real source ICP
        # is copied for coordinate readback only; synthetic meshes are labelled.
        original=Path('/data5/take/brainode/mesh_candidates/adni1_lv_left_qc_v1/ADNI_RID_352_390ec5e9/34537.npz')
        with np.load(original) as a: geometry={k:a[k] for k in ('icp_matrix','centroid_header_world','affine')}
        directory=p['mesh']/subject_folder('ADNI:RID:352');directory.mkdir(parents=True)
        for index in (4,5): np.savez_compressed(directory/(uids[index]+'.npz'),vertices=reconstructed[index],faces=faces,**geometry)
        write_json(directory/'manifest.json',dict(FUNCTIONAL_TEST_ONLY=True))
    return c


def configuration(c,side,continuous=False):
    p=paths(c,side)
    recipe=json.loads((HERE/'assets'/('continuous_recipe.json' if continuous else 'observed_recipe.json')).read_text())
    recipe.update(experiment='FUNCTIONAL_TEST_ONLY',data_root=str(p['data']),
        output=str(p['continuous'] if continuous else p['initial']),report=str(p['report']/('continuous' if continuous else 'observed')),
        code_sha256={name:digest(HERE/name) for name in (CONT_FILES if continuous else TRAIN_FILES)},
        data_sha256={name:digest(p['data']/name) for name in ('observations.npz','pca.npz','summary.json')},test_future_targets=2)
    if continuous:
        recipe.update(initial_checkpoint=str(p['initial']/'latest.pt'),initial_checkpoint_sha256=digest(p['initial']/'latest.pt'),
            voxel_manifest=str(p['voxels']/'manifest.json'),voxel_manifest_sha256=digest(p['voxels']/'manifest.json'),
            grid=json.loads((p['report']/'grid.json').read_text())['grid'],conversion_pairs=1,pseudo_fractions=[.5],progressive_steps=2)
        recipe['cnn'].update(epochs=1,batch_size=2);recipe['ode'].update(epochs=1)
    else: recipe.update(epochs=1,train_temporal_subjects=2,checkpoint_every_epochs=1)
    config=p['report']/('continuous_config.json' if continuous else 'observed_config.json');write_json(config,recipe)
    return config


def run(c,side,stage,resume=False):
    import subprocess
    p=paths(c,side);continuous=stage!='observed';config=p['report']/('continuous_config.json' if continuous else 'observed_config.json')
    script='train_continuous.py' if continuous else 'train_lab.py'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(c['gpus'][SIDES.index(side)]),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,str(HERE/script)]+([stage] if continuous else [])+['--config',str(config)]
    with (p['report']/(stage+('_resume' if resume else '')+'.log')).open('w') as file:
        result=subprocess.run(command,env=env,stdout=file,stderr=subprocess.STDOUT,cwd=HERE)
    if result.returncode: raise RuntimeError(f'functional stage failed: {side} {stage}; see {p["report"]}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--heavy',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();args.report.mkdir(parents=True,exist_ok=False)
    c=setup(args.heavy,args.report);write_json(args.report/'functional_config.json',c)
    for side in SIDES:
        configuration(c,side);run(c,side,'observed')
        initial_hash=digest(paths(c,side)['initial']/'latest.pt');run(c,side,'observed',True)
        assert initial_hash==digest(paths(c,side)['initial']/'latest.pt')
        configuration(c,side,True)
        for stage in ('pseudo','cnn','ode','evaluate'):
            run(c,side,stage)
            if stage in ('cnn','ode'):
                checkpoint=paths(c,side)['continuous']/stage/'latest.pt';before=digest(checkpoint)
                run(c,side,stage,True);assert digest(checkpoint)==before
        print('functional chain passed',side,flush=True)
    from final_export import export_all
    export_all(c)
    # Explicitly label all synthetic artifacts, including download contents.
    root=Path(c['bundle']);(root/'FUNCTIONAL_TEST_ONLY.txt').write_text('Synthetic coefficient integration test. Not measured LV forecasts or scientific accuracy.\n')
    write_json(args.report/'verification.json',dict(status='passed',FUNCTIONAL_TEST_ONLY=True,
        actual_LV_vertices=2490,faces=4976,pitch_mm=1.5,chain='observed→voxel→Eq4→CNN→augmented ODE→Algorithm2→3D export',
        epoch_resume_reached_fixed_final=True,checkpoint_hashes_unchanged_on_final_epoch_restore=True))


if __name__=='__main__':main()
