"""Continue a verified engineering prefix using the prior fitted mesh.

The only size initialization is in the already fitted first observation.
Later observations use previous vertices unchanged, as in BrainODE A.1.
"""

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys

import nibabel as nib
import numpy as np
import open3d as o3d
import torch
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_fit import (FITTING_RECIPE_VERSION, PROJECT, mask_overlap, reconstruct_scan,
                      rigid_target_to_source, sha256, surface_metrics, write_obj)
from mesh_input import boundary_geometry


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',required=True)
    args=parser.parse_args()
    prefix=args.prefix.resolve()
    output=args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('QC outputs must stay in the home project')
    prior=json.loads((prefix/'manifest.json').read_text())
    if (prior['fitting_recipe_version']!=FITTING_RECIPE_VERSION
        or prior['fitting_code_sha256']!=sha256(PROJECT/'mesh_fit.py')
        or prior['boundary_code_sha256']!=sha256(PROJECT/'mesh_input.py')
        or len(prior['scans'])!=1
        or prior['initialization']!='mask-volume'
        or prior['update_mode']!='fixed-source' or prior['edge_mode']!='public'):
        raise ValueError('prefix recipe/code differs from this continuation')
    first=copy.deepcopy(prior['scans'][0])
    if first['cohort']!='ADNI1':
        parser.error('this engineering series check uses the existing ADNI1 scan table')
    with (PROJECT/'artifacts/adni1_allage_eligibility/eligible.csv').open() as file:
        rows=sorted((r for r in csv.DictReader(file) if r['pid']==first['pid']),
                    key=lambda r:(r['day'],r['uid']))
    if not rows or rows[0]['uid']!=first['uid'] or len({r['day'] for r in rows})!=len(rows):
        raise ValueError('prefix is not the first observation of this no-repeat QC series')
    if (sha256(Path(rows[0]['parcellation_path']))!=first['source_sha256']
        or float(rows[0]['age'])!=float(first['age']) or rows[0]['class']!=first['diagnosis']):
        raise ValueError('prefix source/metadata mismatch')
    template_path=PROJECT/'dataset/Shapes/hippocampus_template_mesh.obj'
    if sha256(template_path)!=prior['template_sha256']:
        raise ValueError('template changed')
    template=trimesh.load_mesh(template_path,process=False)
    with np.load(prefix/f"{first['uid']}.npz") as data:
        faces=data['faces'].copy()
        vertices=data['vertices'].copy()
        np.testing.assert_array_equal(faces,template.faces)
        np.testing.assert_array_equal(data['unscaled_template_vertices'],template.vertices)
        pivot=template.vertices.mean(0)
        expected=(template.vertices-pivot)*first['initialization_factor']+pivot
        np.testing.assert_array_equal(data['source_vertices'],expected)
    output.mkdir(parents=True,exist_ok=False)
    for suffix in ('.npz','.obj','_loss.csv'):
        shutil.copy2(prefix/f"{first['uid']}{suffix}",output/f"{first['uid']}{suffix}")
    first.update(day=rows[0]['day'],age=float(first['age']),source_uid='template')
    manifest=copy.deepcopy(prior)
    manifest.update(device=args.device,initialization_applies_to='first observation only',
        reused_prefix=dict(directory=str(prefix),manifest_sha256=sha256(prefix/'manifest.json'),scans=1),
        scans=[first])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.set_num_threads(4)
    o3d.utility.random.seed(prior['seed'])
    for index,row in enumerate(rows[1:],start=1):
        path=Path(row['parcellation_path'])
        source_hash=sha256(path)
        image=nib.load(path)
        mask=np.asarray(image.dataobj)==75
        geometry=boundary_geometry(mask,image.affine)
        source=vertices.copy()
        target,matrix,icp=rigid_target_to_source(geometry['target_centered_header_world_units'],source)
        before=surface_metrics(source,faces,target)
        vertices,elapsed=reconstruct_scan(source,faces,target,prior['iterations'],args.device,
            prior['seed']+index,output/f"{row['uid']}_loss.csv",'fixed-source','public')
        after=surface_metrics(vertices,faces,target)
        mask_volume=float(mask.sum()*abs(np.linalg.det(image.affine[:3,:3])))
        record=dict(cohort='ADNI1',uid=row['uid'],pid=row['pid'],day=row['day'],age=float(row['age']),
            diagnosis=row['class'],source_uid=rows[index-1]['uid'],source_path=str(path),
            source_sha256=source_hash,source_spatial_units=image.header.get_xyzt_units()[0],
            mask_volume_mm3=mask_volume,boundary_points=len(target),
            initialization='previous fitted mesh, no additional scaling',initialization_factor=1.,
            before=before,after=after,overlap=mask_overlap(vertices,faces,geometry,matrix,mask),
            volume_bias_percent=100*(after['volume_mm3']/mask_volume-1),icp=icp,
            elapsed_seconds=elapsed,status='engineering_QC_candidate')
        np.savez_compressed(output/f"{row['uid']}.npz",vertices=vertices,faces=faces,source_vertices=source,
            target=target,icp_matrix=matrix,centroid_header_world=geometry['centroid_header_world'],affine=geometry['affine'])
        write_obj(output/f"{row['uid']}.obj",vertices,faces)
        if sha256(path)!=source_hash:
            raise RuntimeError('source changed during read-only QC')
        manifest['scans'].append(record)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
