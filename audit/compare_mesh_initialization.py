"""Engineering QC of template size for existing label-75 masks.

The mask and its physical size remain unchanged. Mask-volume initialization
only changes the starting template, preserving vertex/face indices. This is
an explicit, unreported-in-the-paper supplement, not an adopted cohort rule.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import open3d as o3d
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh_fit import (FITTING_RECIPE_VERSION, PROJECT, WEIGHTS, mask_overlap,
                      reconstruct_scan, rigid_target_to_source, sha256,
                      surface_metrics, write_obj)
from mesh_input import boundary_geometry


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--initialization', choices=['unscaled', 'mask-volume'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261006)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('lightweight engineering QC outputs must stay in the project')
    table = PROJECT / 'artifacts/multicohort_inventory/all_observations.csv'
    with table.open() as file:
        rows = [r for r in csv.DictReader(file) if r['cohort']==args.cohort and r['uid']==args.uid]
    if len(rows) != 1:
        parser.error('cohort + UID must identify one source observation')
    row, = rows
    path = Path(row['parcellation_path'])
    source_hash = sha256(path)
    image = nib.load(path)
    mask = np.asarray(image.dataobj) == 75
    geometry = boundary_geometry(mask, image.affine)
    template_path = PROJECT / 'dataset/Shapes/hippocampus_template_mesh.obj'
    template = trimesh.load_mesh(template_path, process=False)
    original, faces = np.asarray(template.vertices).copy(), np.asarray(template.faces).copy()
    if original.shape != (1490,3) or faces.shape != (2976,3) or template.volume <= 0:
        raise ValueError('unexpected template topology or signed volume')
    mask_volume = float(mask.sum()*abs(np.linalg.det(image.affine[:3,:3])))
    factor = float((mask_volume/template.volume)**(1/3)) if args.initialization=='mask-volume' else 1.
    pivot = original.mean(0)
    source = (original-pivot)*factor+pivot
    initial_volume = float(trimesh.Trimesh(source, faces, process=False).volume)
    if not np.isclose(initial_volume, template.volume*factor**3):
        raise ValueError('initialization did not preserve the expected volume')
    torch.set_num_threads(4)
    o3d.utility.random.seed(args.seed)
    target, transform, icp = rigid_target_to_source(
        geometry['target_centered_header_world_units'], source)
    output.mkdir(parents=True, exist_ok=False)
    vertices, elapsed = reconstruct_scan(source, faces, target, args.iterations,
        args.device, args.seed, output/f'{args.uid}_loss.csv', 'fixed-source', 'public')
    metrics = surface_metrics(vertices, faces, target)
    overlap = mask_overlap(vertices, faces, geometry, transform, mask)
    np.savez_compressed(output/f'{args.uid}.npz', vertices=vertices,faces=faces,
                       source_vertices=source,unscaled_template_vertices=original,
                       target=target,icp_matrix=transform,
                       centroid_header_world=geometry['centroid_header_world'],affine=geometry['affine'])
    write_obj(output/f'{args.uid}.obj', vertices, faces)
    reloaded = trimesh.load_mesh(output/f'{args.uid}.obj', process=False)
    np.testing.assert_allclose(reloaded.vertices, vertices, atol=1e-7, rtol=1e-8)
    np.testing.assert_array_equal(reloaded.faces, faces)
    if sha256(path)!=source_hash:
        raise RuntimeError('source mask changed during read-only QC')
    record = dict(cohort=args.cohort,uid=args.uid,pid=row['pid'],age=row['age'],
                  diagnosis=row['diagnosis'],source_path=str(path),source_sha256=source_hash,
                  source_spatial_units=image.header.get_xyzt_units()[0],mask_volume_mm3=mask_volume,
                  boundary_points=len(target),initialization=args.initialization,
                  initialization_factor=factor,initial_template_volume_mm3=initial_volume,
                  before=surface_metrics(source,faces,target),after=metrics,overlap=overlap,
                  volume_bias_percent=100*(metrics['volume_mm3']/mask_volume-1),
                  icp=icp,elapsed_seconds=elapsed,status='engineering_QC_candidate')
    manifest = dict(fitting_recipe_version=FITTING_RECIPE_VERSION,qc_example=True,
                    scientific_training_adoption=False,initialization=args.initialization,
                    supplement='template-only size initialization; no mask or final-volume normalization',
                    subject=row['pid'],iterations=args.iterations,seed=args.seed,device=args.device,
                    update_mode='fixed-source',edge_mode='public',weights=WEIGHTS,
                    template_sha256=sha256(template_path),
                    face_array_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
                    fitting_code_sha256=sha256(PROJECT/'mesh_fit.py'),
                    boundary_code_sha256=sha256(PROJECT/'mesh_input.py'),
                    pointnet_code_sha256=sha256(PROJECT/'references/BrainSubcorticalStructure/models/pointnet.py'),
                    spatial_unit_assumption='1 numeric unit = 1 mm',
                    final_output='after final optimizer update, final iteration input, train-mode BatchNorm',
                    loss_log='per-iteration loss before that optimizer update',scans=[record])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
