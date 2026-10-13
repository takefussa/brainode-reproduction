"""Controlled first-scan QC using the author's unchanged loss function.

Loads only criterion's AST to bypass an unavailable, unused upstream import
of models.point_mesh_loss2. The function itself and upstream edge loss are
unchanged. This diagnostic does not adopt a different scientific recipe.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
from pytorch3d import loss as p3d_loss
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh_fit import FITTING_RECIPE_VERSION, PROJECT, mask_overlap, reconstruct_scan, sha256, surface_metrics, write_obj
from mesh_input import boundary_geometry


def load_author_criterion():
    root = PROJECT / 'references/BrainSubcorticalStructure/models'
    spec = importlib.util.spec_from_file_location('diagnostic_author_edge', root / 'loss_3d.py')
    edge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edge)
    tree = ast.parse((root / 'loss.py').read_text())
    function, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'criterion']
    namespace = dict(Meshes=Meshes, Pointclouds=Pointclouds,
                     sample_points_from_meshes=sample_points_from_meshes,
                     mesh_edge_var_loss=edge.mesh_edge_var_loss)
    for name in ('chamfer_distance', 'mesh_laplacian_smoothing', 'mesh_normal_consistency',
                 'point_mesh_face_distance'):
        namespace[name] = getattr(p3d_loss, name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(root / 'loss.py'), 'exec'), namespace)
    ast_hash = hashlib.sha256(ast.dump(function).encode()).hexdigest()
    return namespace['criterion'], ast_hash


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path,
                        default=PROJECT / 'artifacts/mesh_qc/compare_fixed_publicedge5000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('QC outputs must stay in the home project')
    baseline = json.loads((args.baseline / 'manifest.json').read_text())
    if len(baseline['scans']) != 1 or baseline['update_mode'] != 'fixed-source' or baseline['edge_mode'] != 'public':
        parser.error('requires the single-scan fixed-source/public-edge comparison')
    record = copy.deepcopy(baseline['scans'][0])
    uid = record['uid']
    if sha256(Path(record['source_path'])) != record['source_sha256']:
        raise ValueError('source mask changed')
    with np.load(args.baseline / f'{uid}.npz') as cached:
        arrays = {key: cached[key].copy() for key in cached.files}
    criterion, ast_hash = load_author_criterion()

    def public_loss(mesh, target, reference_vertices, reference_normals, edge_mode):
        value, log = criterion(mesh.verts_packed()[None], target[None], mesh.faces_packed()[None])
        vertex = torch.nn.functional.mse_loss(mesh.verts_packed(), reference_vertices)
        terms = dict(chamfer=log['chamfer_loss'], point_mesh=log['point_mesh_dist_loss'],
                     vertex=vertex, normal=vertex.new_zeros(()), edge=log['edge_loss'],
                     consistency=log['normal_consistency_loss'], laplacian=log['laplacian_loss'])
        return value + vertex, terms

    output.mkdir(parents=True, exist_ok=False)
    vertices, elapsed = reconstruct_scan(
        arrays['source_vertices'], arrays['faces'], arrays['target'], baseline['iterations'],
        args.device, baseline['seed'], output / f'{uid}_loss.csv', 'fixed-source', 'public',
        loss_function=public_loss)
    arrays['vertices'] = vertices
    np.savez(output / f'{uid}.npz', **arrays)
    write_obj(output / f'{uid}.obj', vertices, arrays['faces'])
    image = nib.load(record['source_path'])
    mask = np.asarray(image.dataobj) == 75
    geometry = boundary_geometry(mask, image.affine)
    record['after'] = surface_metrics(vertices, arrays['faces'], arrays['target'])
    record['overlap'] = mask_overlap(vertices, arrays['faces'], geometry, arrays['icp_matrix'], mask)
    record['elapsed_seconds'] = elapsed
    record['status'] = 'diagnostic_candidate_requires_quality_review'
    manifest = copy.deepcopy(baseline)
    manifest.update(fitting_recipe_version=FITTING_RECIPE_VERSION,
                    fitting_code_sha256=sha256(PROJECT/'mesh_fit.py'),
                    final_output='after final optimizer update, final iteration input, train-mode BatchNorm',
                    device=args.device, scans=[record], recipe_kind='diagnostic_author_loss',
                    scientific_training_adoption=False,
                    diagnostic_difference='public coordinate MSE instead of vertex RMS; no normal term',
                    baseline_directory=str(args.baseline.resolve()),
                    baseline_manifest_sha256=sha256(args.baseline / 'manifest.json'),
                    author_loss_sha256=sha256(PROJECT / 'references/BrainSubcorticalStructure/models/loss.py'),
                    author_edge_sha256=sha256(PROJECT / 'references/BrainSubcorticalStructure/models/loss_3d.py'),
                    criterion_ast_sha256=ast_hash,
                    compatibility='execute unchanged criterion AST; bypass unused missing point_mesh_loss2 import')
    manifest['weights']['normal'] = 0.
    manifest['paper_supplements']['displacement_reduction'] = 'public coordinate MSE'
    manifest['paper_supplements']['normals'] = 'omitted as in public source'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
