"""Fixed-topology LV fitting using the existing BrainODE A.1 mesh losses.

LV-Net supplies the template. Its joint-anatomy loss is not silently substituted
for the seven-loss BrainODE reconstruction recipe. Region masks retain all
selected-label voxels and connected components.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import open3d as o3d
from scipy import ndimage
import torch
import trimesh

import mesh_fit as core
from mesh_cohort import ORDERING_POLICY, QC_THRESHOLDS, ordered_scans, review_flags
from mesh_initialization import initialize_template
from mesh_input import boundary_geometry

PROJECT = Path(__file__).resolve().parent
LABELS = {'left': (163,164,165,166,167), 'right': (168,169,170,171,172)}


def main(argv=None, *, on_scan_done=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--side', choices=LABELS, required=True)
    parser.add_argument('--scan-table', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--template-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--max-scans', type=int, help='Engineering QC prefix only; never a cohort exclusion')
    parser.add_argument('--scope', choices=['engineering-qc','cohort-candidates'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.iterations < 1 or (args.max_scans is not None and args.max_scans < 1):
        parser.error('iterations and max-scans must be positive')
    if args.scope == 'cohort-candidates' and args.max_scans is not None:
        parser.error('cohort mode must retain every observation')
    output = args.output.resolve()
    if not (output.is_relative_to(Path('/home/take/brainode'))
            or output.is_relative_to(Path('/data5/take/brainode'))):
        parser.error('derived outputs must remain under home project or data5')
    with args.scan_table.open() as file:
        scans = ordered_scans(row for row in csv.DictReader(file) if row['pid'] == args.subject)
    table_count = len(scans)
    if args.max_scans is not None:
        scans = scans[:args.max_scans]
    audit = json.loads(args.template_audit.read_text())
    expected = audit['assets'][args.side]
    if core.sha256(args.template) != expected['sha256'][args.template.name]:
        raise ValueError('derived template does not match its preparation audit')
    with np.load(args.template, allow_pickle=False) as data:
        template_vertices, faces = data['vertices'].copy(), data['faces'].copy()
    surface = trimesh.Trimesh(template_vertices, faces, process=False)
    if not surface.is_watertight or not surface.is_winding_consistent or surface.volume <= 0:
        raise ValueError('invalid LV template')
    first = nib.load(scans[0]['parcellation_path'])
    first_mask = np.isin(np.asarray(first.dataobj), LABELS[args.side])
    first_volume = float(first_mask.sum()*abs(np.linalg.det(first.affine[:3,:3])))
    vertices, initialization = initialize_template(template_vertices, faces,
        mode='mask-volume', mask_volume=first_volume)
    provenance = core.code_provenance()
    provenance['lv_mesh_fit.py'] = core.sha256(Path(__file__))
    manifest = dict(fitting_recipe_version=core.FITTING_RECIPE_VERSION,
        region_recipe_version=1, region='lateral_ventricle_'+args.side,
        labels=list(LABELS[args.side]), template_audit_sha256=core.sha256(args.template_audit),
        original_template_sha256=audit['source_template_sha256'],
        initialization='mask-volume', initialization_info=initialization,
        subject=args.subject, qc_example=args.scope=='engineering-qc',
        scientific_training_adoption=False, spatial_unit_assumption='1 numeric unit = 1 mm',
        iterations=args.iterations, seed=args.seed, device=args.device, weights=core.WEIGHTS,
        physical_gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        update_mode='fixed-source', edge_mode='public', scan_count_in_table=table_count,
        observation_ordering=ORDERING_POLICY, same_day_repeat_policy='retain all; previous fitted observation is next source',
        qc_thresholds=QC_THRESHOLDS, qc_flags_are_exclusion_rules=False,
        scan_table_sha256=core.sha256(args.scan_table), code_provenance=provenance,
        environment=dict(torch=str(torch.__version__), numpy=np.__version__, open3d=o3d.__version__,
            cuda=torch.version.cuda, pytorch3d=importlib.metadata.version('pytorch3d'),
            scipy=importlib.metadata.version('scipy'),python=sys.version),
        template_sha256=core.sha256(args.template), face_array_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
        final_output='recomputed after final optimizer update with final parameters',
        reconstruction_method='BrainODE Appendix A.1 seven-loss PointNet recipe reused unchanged; LV-Net template only',
        public_code_difference='No LV-Net joint hippocampal/texture losses or anisotropic learned template scaling',
        voxel_policy='all cavity-label voxels and components retained; no mask smoothing, filling, registration resampling or scaling',
        scans=[])
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(4)
    o3d.utility.random.seed(args.seed)
    start = 0
    if args.resume and (output/'manifest.json').exists():
        saved = json.loads((output/'manifest.json').read_text())
        for key in ['region_recipe_version','region','labels','template_audit_sha256','original_template_sha256']:
            if saved.get(key) != manifest[key]:
                raise ValueError('cached LV setting differs: '+key)
        vertices, start = core.reuse_prefix(output,output,manifest,scans,vertices,faces)
    elif any(output.iterdir()):
        raise ValueError('nonempty LV output has no resumable manifest')
    core.atomic_manifest(output/'manifest.json',manifest)
    for index in range(start,len(scans)):
        row = scans[index]
        uid = row['uid']
        path = Path(row['parcellation_path'])
        source_sha = core.sha256(path)
        image = nib.load(path)
        mask = np.isin(np.asarray(image.dataobj),LABELS[args.side])
        geometry = boundary_geometry(mask,image.affine)
        mask_points = np.argwhere(mask)
        _, components = ndimage.label(mask,structure=np.ones((3,3,3),dtype=np.uint8))
        mask_volume = float(mask.sum()*abs(np.linalg.det(image.affine[:3,:3])))
        target, transform, icp = core.rigid_target_to_source(
            geometry['target_centered_header_world_units'],vertices)
        source = vertices.copy()
        before = core.surface_metrics(source,faces,target)
        print(f"SCAN {index+1}/{len(scans)} {args.side} UID={uid}; vertices={len(vertices)} boundary={len(target)}",flush=True)
        vertices, elapsed = core.reconstruct_scan(source,faces,target,args.iterations,args.device,
            args.seed+index,output/f'{uid}_loss.csv','fixed-source','public')
        after = core.surface_metrics(vertices,faces,target)
        overlap = core.mask_overlap(vertices,faces,geometry,transform,mask)
        record = dict(uid=uid,day=row['day'],age=float(row['age']),diagnosis=row['class'],
            cohort=row.get('cohort',''),scan_key=row.get('scan_key',uid),split=row.get('split',''),
            region=manifest['region'],labels=manifest['labels'],source_uid=scans[index-1]['uid'] if index else 'template',
            initialization='mask-volume' if index==0 else 'previous fitted mesh, no additional scaling',
            initialization_factor=initialization['factor'] if index==0 else 1.,
            source_path=str(path),source_sha256=source_sha,source_spatial_units=image.header.get_xyzt_units()[0],
            mask_volume_mm3=mask_volume,mask_components_26=int(components),
            mask_touches_image_edge=bool(np.any(mask_points.min(0)==0) or np.any(mask_points.max(0)==np.array(mask.shape)-1)),
            mask_bbox_voxel=[mask_points.min(0).tolist(),mask_points.max(0).tolist()],
            volume_bias_percent=float(100*(after['volume_mm3']/mask_volume-1)),
            boundary_points=len(target),before=before,after=after,overlap=overlap,icp=icp,
            elapsed_seconds=elapsed,status='candidate_requires_quality_review')
        record['qc_flags'] = review_flags(record)
        record['previous_scan_qc_flags'] = manifest['scans'][-1].get('qc_flags',[]) if index else []
        if core.sha256(path) != source_sha:
            raise ValueError('source image changed during fitting')
        with (output/f'{uid}.npz.tmp').open('wb') as file:
            np.savez_compressed(file,vertices=vertices,faces=faces,source_vertices=source,
                target=target,icp_matrix=transform,centroid_header_world=geometry['centroid_header_world'],affine=geometry['affine'])
        os.replace(output/f'{uid}.npz.tmp',output/f'{uid}.npz')
        core.write_obj(output/f'{uid}.obj.tmp',vertices,faces)
        os.replace(output/f'{uid}.obj.tmp',output/f'{uid}.obj')
        record['artifact_sha256'] = {uid+s:core.sha256(output/(uid+s)) for s in ['.npz','.obj','_loss.csv']}
        manifest['scans'].append(record)
        core.atomic_manifest(output/'manifest.json',manifest)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if on_scan_done is not None:
            on_scan_done(record)


if __name__=='__main__':
    main()
