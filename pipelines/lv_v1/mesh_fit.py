"""Paper-based, fixed-topology longitudinal hippocampal mesh fitting.

The seven loss weights and longitudinal updates follow BrainODE A.1.
Unspecified reductions are explicit supplements, not recovered author code:
vertex/normal RMS L2 and area-weighted vertex normals against the scan's
initial mesh. The adopted edge mode uses the public expression; the literal
edge variance remains available for diagnostics. No ICV scaling is applied.
"""

import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import nibabel as nib
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy import ndimage
from pytorch3d.loss import (
    chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency,
    point_mesh_face_distance,
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds

from mesh_input import boundary_geometry
from mesh_initialization import initialize_template
from mesh_cohort import ORDERING_POLICY, QC_THRESHOLDS, ordered_scans, review_flags

PROJECT = Path(__file__).resolve().parent
WORKSPACE = Path('/home/take/brainode')
FITTING_RECIPE_VERSION = 3  # Explicit repeat ordering and verified resumable artifacts.
WEIGHTS = dict(chamfer=0.5, point_mesh=3., vertex=1., normal=1.,
               edge=1500., consistency=1., laplacian=5.)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_provenance():
    paths = [PROJECT/name for name in ('mesh_fit.py', 'mesh_input.py', 'mesh_initialization.py', 'mesh_cohort.py')]
    paths.append(PROJECT/'references/BrainSubcorticalStructure/models/pointnet.py')
    return {str(path.relative_to(PROJECT)): sha256(path) for path in paths}


def atomic_manifest(path, manifest):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def rms_l2(difference):
    # Smooth the origin without a nonzero penalty there; epsilon is numerical.
    return torch.sqrt(difference.square().sum(-1).mean() + 1e-12) - 1e-6


def paper_regularizers(mesh, reference_vertices, reference_normals, edge_mode='paper-variance'):
    vertices = mesh.verts_packed()
    edges = mesh.edges_packed()
    edge_lengths = (vertices[edges[:, 0]] - vertices[edges[:, 1]]).norm(dim=-1)
    if edge_mode == 'paper-variance':
        edge_loss = edge_lengths.var(unbiased=False)
    elif edge_mode == 'public':
        # Exact public loss_3d.py expression for one mesh, target_length=0.
        edge_loss = (edge_lengths.square() / len(edge_lengths)).std(unbiased=True)
    else:
        raise ValueError('unknown edge loss mode')
    return dict(
        vertex=rms_l2(vertices - reference_vertices),
        normal=rms_l2(mesh.verts_normals_packed() - reference_normals),
        edge=edge_loss,
        consistency=mesh_normal_consistency(mesh),
        laplacian=mesh_laplacian_smoothing(mesh, method='uniform'),
    )


def fitting_loss(mesh, target, reference_vertices, reference_normals, edge_mode):
    terms = paper_regularizers(mesh, reference_vertices, reference_normals, edge_mode)
    # Follow the public mesh-to-surface sampling convention. The sampling count
    # is the number of boundary points, not a downsampling of the target mask.
    sampled = sample_points_from_meshes(mesh, target.shape[0])
    terms['chamfer'] = chamfer_distance(sampled, target[None])[0]
    # The upstream PyTorch3D function includes both point->face and face->point;
    # Park §2.3 explicitly describes this bidirectional distance.
    terms['point_mesh'] = point_mesh_face_distance(mesh, Pointclouds([target]))
    total = sum(WEIGHTS[key] * value for key, value in terms.items())
    return total, terms


def rigid_target_to_source(target, source_vertices):
    moving = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    fixed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_vertices))
    result = o3d.pipelines.registration.registration_icp(
        moving, fixed, 10., np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000),
    )
    matrix = np.asarray(result.transformation)
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6):
        raise RuntimeError('ICP introduced a non-rigid transform')
    aligned = target @ matrix[:3, :3].T + matrix[:3, 3]
    return aligned, matrix, dict(fitness=result.fitness, inlier_rmse=result.inlier_rmse)


def official_pointnet(device):
    path = PROJECT / 'references/BrainSubcorticalStructure/models/pointnet.py'
    spec = importlib.util.spec_from_file_location('official_mesh_pointnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PointNetOpt(3, input_transform=False, feature_transform=False).to(device)


def reconstruct_scan(source_vertices, faces, target, iterations, device, seed, log_path,
                     update_mode, edge_mode, *, loss_function=None):
    loss_function = fitting_loss if loss_function is None else loss_function
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    vertices = torch.tensor(source_vertices, dtype=torch.float32, device=device)
    face_tensor = torch.tensor(faces, dtype=torch.int64, device=device)
    point_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    reference = Meshes([vertices], [face_tensor])
    reference_vertices = vertices.clone().detach()
    reference_normals = reference.verts_normals_packed().clone().detach()
    model = official_pointnet(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    # Inherited from the cited mesh method, not BrainODE's ODE training epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    start = time.monotonic()
    with log_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['iteration', 'total', *WEIGHTS, 'seconds'])
        writer.writeheader()
        for iteration in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            input_vertices = vertices if update_mode == 'accumulated' else reference_vertices
            displacement = model(input_vertices[None]).transpose(1, 2)[0]
            updated = input_vertices + displacement
            mesh = Meshes([updated], [face_tensor])
            loss, terms = loss_function(mesh, point_tensor, reference_vertices, reference_normals, edge_mode)
            if not torch.isfinite(loss) or not torch.isfinite(updated).all():
                raise RuntimeError(f'nonfinite mesh/loss at iteration {iteration}; inspect log')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f'nonfinite gradient at iteration {iteration}; inspect log')
            optimizer.step()
            scheduler.step()
            values = dict(iteration=iteration + 1, total=float(loss.detach()),
                          **{key: float(value.detach()) for key, value in terms.items()},
                          seconds=time.monotonic() - start)
            writer.writerow(values)
            # Truncate the optimization graph. In fixed-source mode, the next
            # optimizer iteration still starts from reference_vertices.
            vertices = updated.detach()
            if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
                file.flush()
                print(json.dumps(values), flush=True)
    # Each logged loss is evaluated before its optimizer update. Recompute
    # geometry with the final parameters, using the last iteration's input and
    # the same train-mode BatchNorm convention as during fitting. Returning
    # `updated` above would discard the effect of the final optimizer update.
    with torch.no_grad():
        vertices = input_vertices + model(input_vertices[None]).transpose(1, 2)[0]
    if not torch.isfinite(vertices).all():
        raise RuntimeError('nonfinite final mesh after optimizer update')
    return vertices.cpu().numpy(), time.monotonic() - start


def surface_metrics(vertices, faces, target):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = scene.compute_distance(o3d.core.Tensor(target.astype(np.float32))).numpy()
    shape = trimesh.Trimesh(vertices, faces, process=False)
    intersections = np.asarray(mesh.get_self_intersecting_triangles())
    return dict(boundary_to_surface_mean_mm=float(distances.mean()),
                boundary_to_surface_p95_mm=float(np.percentile(distances, 95)),
                boundary_to_surface_max_mm=float(distances.max()),
                volume_mm3=float(shape.volume), area_mm2=float(shape.area),
                edge_length_std_mm=float(shape.edges_unique_length.std()),
                min_triangle_area_mm2=float(shape.area_faces.min()),
                self_intersecting_triangle_pairs=len(intersections),
                triangles_in_self_intersections=len(np.unique(intersections)) if intersections.size else 0,
                watertight=bool(shape.is_watertight), winding_consistent=bool(shape.is_winding_consistent))


def mask_overlap(vertices, faces, geometry, icp_matrix, mask):
    inverse = np.linalg.inv(icp_matrix)
    native_world = (vertices @ inverse[:3, :3].T + inverse[:3, 3]
                    + geometry['centroid_header_world'])
    voxel = nib.affines.apply_affine(np.linalg.inv(geometry['affine']), native_world)
    mask_points = np.argwhere(mask)
    lower = np.floor(np.minimum(voxel.min(0), mask_points.min(0))).astype(int) - 1
    upper = np.ceil(np.maximum(voxel.max(0), mask_points.max(0))).astype(int) + 1
    size = upper - lower + 1
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(voxel),
                                   o3d.utility.Vector3iVector(faces))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # LV bounding boxes are larger than hippocampal boxes. Visit every voxel
    # in bounded slabs, retaining the identical center-occupancy definition.
    intersection = reconstructed = target_count = 0
    yz = np.indices(tuple(size[1:])).reshape(2, -1).T + lower[1:]
    for x in range(int(lower[0]), int(upper[0])+1):
        queries = np.column_stack([np.full(len(yz), x), yz]).astype(np.float32)
        inside = scene.compute_occupancy(o3d.core.Tensor(queries), nsamples=3).numpy() > 0.5
        indices = queries.astype(int)
        in_image = ((indices >= 0) & (indices < np.array(mask.shape))).all(1)
        target = np.zeros(len(indices), dtype=bool)
        target[in_image] = mask[tuple(indices[in_image].T)]
        intersection += int(np.count_nonzero(inside & target))
        reconstructed += int(inside.sum())
        target_count += int(target.sum())
    if target_count != int(mask.sum()):
        raise ValueError('native-grid Dice bounding box missed mask voxels')
    return dict(dice=float(2*intersection/(reconstructed+target_count)),
                reconstructed_voxels=reconstructed, mask_voxels=target_count,
                query_policy='all voxel centers in complete bounding box; x slabs')


def write_obj(path, vertices, faces):
    # Explicit writing preserves vertex order and face winding.
    with path.open('w') as file:
        for x, y, z in vertices:
            file.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
        for a, b, c in faces + 1:
            file.write(f'f {a} {b} {c}\n')


def reuse_prefix(directory, output, manifest, scans, template_vertices, faces):
    """Reuse a verified chronological prefix, never mix fitting recipes."""
    cached = json.loads((directory / 'manifest.json').read_text())
    if not cached.get('code_provenance') or not manifest.get('code_provenance'):
        raise ValueError('cached fitting code_provenance is missing')
    for key in ('fitting_recipe_version', 'initialization', 'initialization_info',
                'subject', 'iterations', 'seed', 'update_mode', 'edge_mode',
                'weights', 'template_sha256', 'face_array_sha256', 'spatial_unit_assumption',
                'code_provenance', 'environment', 'observation_ordering', 'scan_table_sha256'):
        if cached.get(key) != manifest.get(key):
            raise ValueError(f'cached fitting setting differs: {key}')
    prefix = cached['scans']
    if len(prefix) > len(scans):
        raise ValueError('cached series is longer than the requested observations')
    vertices = template_vertices.copy()
    for index, record in enumerate(prefix):
        row = scans[index]
        if (record['uid'], record['day'], record['age'], record['diagnosis']) != (
                row['uid'], row['day'], float(row['age']), row['class']):
            raise ValueError('cached observations are not the requested chronological prefix')
        if sha256(Path(row['parcellation_path'])) != record['source_sha256']:
            raise ValueError('cached mask hash differs from the current source')
        expected_files = {record['uid']+suffix for suffix in ('.npz', '.obj', '_loss.csv')}
        if set(record.get('artifact_sha256', {})) != expected_files:
            raise ValueError('cached artifact hash set is incomplete')
        for filename, digest in record['artifact_sha256'].items():
            if Path(filename).name != filename or sha256(directory/filename) != digest:
                raise ValueError('cached artifact hash differs')
        with np.load(directory / f"{row['uid']}.npz") as data:
            if not np.array_equal(data['faces'], faces) or not np.array_equal(data['source_vertices'], vertices):
                raise ValueError('cached topology or longitudinal source differs')
            candidate = data['vertices'].copy()
            if candidate.shape != template_vertices.shape or not np.isfinite(candidate).all():
                raise ValueError('invalid cached vertices')
        vertices = candidate
    # Validate the entire prefix before copying any artifact into the new run.
    for record in prefix:
        for suffix in ('.npz', '.obj', '_loss.csv'):
            filename = record['uid'] + suffix
            if directory.resolve() != output.resolve():
                shutil.copy2(directory / filename, output / filename)
    manifest['scans'] = prefix
    manifest['reused_prefix'] = dict(directory=str(directory), scans=len(prefix),
                                   manifest_sha256=sha256(directory / 'manifest.json'))
    return vertices, len(prefix)


def main(argv=None, *, on_scan_done=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--scan-table', type=Path,
                        default=PROJECT / 'artifacts/adni1_allage_eligibility/eligible.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--max-scans', type=int, help='limit an engineering QC run; not a cohort rule')
    parser.add_argument('--resume-from', type=Path, help='reuse a matching chronological prefix')
    parser.add_argument('--resume', action='store_true', help='resume a verified prefix in the output directory')
    parser.add_argument('--scope', choices=['engineering-qc', 'cohort-candidates'], default='engineering-qc')
    # Keep diagnostic recipes explicit. The adopted method is fixed-source /
    # public; the accumulated interpretation failed actual mesh QC.
    parser.add_argument('--update-mode', choices=['accumulated', 'fixed-source'], required=True)
    parser.add_argument('--edge-mode', choices=['paper-variance', 'public'], required=True)
    parser.add_argument('--template-initialization', choices=['unscaled', 'mask-volume'], required=True,
                        help='first template only; mask-volume is a documented lab-data supplement')
    args = parser.parse_args(argv)
    if args.resume and args.resume_from is not None:
        parser.error('resume and resume-from are mutually exclusive')
    if args.iterations < 1:
        parser.error('iterations must be positive')
    output = args.output.resolve()
    if not output.is_relative_to(WORKSPACE) and not output.is_relative_to(Path('/data5/take/brainode')):
        parser.error('derived outputs must be under the project or its data5 directory')
    with args.scan_table.open() as file:
        scans = ordered_scans(row for row in csv.DictReader(file) if row['pid'] == args.subject)
    scan_count_in_table = len(scans)
    if args.max_scans is not None:
        if args.max_scans < 1:
            parser.error('max-scans must be positive')
        scans = scans[:args.max_scans]
    # This explicit subject is an engineering QC example, not cohort selection.
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(4)
    o3d.utility.random.seed(args.seed)
    template_path = PROJECT / 'dataset/Shapes/hippocampus_template_mesh.obj'
    template = trimesh.load_mesh(template_path, process=False)
    vertices, faces = np.asarray(template.vertices).copy(), np.asarray(template.faces).copy()
    if vertices.shape != (1490, 3) or faces.shape != (2976, 3):
        raise ValueError('unexpected template topology')
    first_mask_volume = None
    if args.template_initialization == 'mask-volume':
        first_image = nib.load(scans[0]['parcellation_path'])
        first_count = np.count_nonzero(np.asarray(first_image.dataobj) == 75)
        first_mask_volume = float(first_count * abs(np.linalg.det(first_image.affine[:3, :3])))
    vertices, initialization_info = initialize_template(
        vertices, faces, mode=args.template_initialization, mask_volume=first_mask_volume)
    manifest = dict(fitting_recipe_version=FITTING_RECIPE_VERSION,
                    initialization=args.template_initialization, initialization_info=initialization_info,
                    final_output='after final optimizer update, using final iteration input and train-mode BatchNorm',
                    loss_log='per-iteration loss before that optimizer update',
                    subject=args.subject, qc_example=args.scope=='engineering-qc',
                    scientific_training_adoption=False,
                    spatial_unit_assumption='1 numeric unit = 1 mm',
                    iterations=args.iterations, seed=args.seed, device=args.device, weights=WEIGHTS,
                    update_mode=args.update_mode, edge_mode=args.edge_mode,
                    scan_count_in_table=scan_count_in_table,
                    observation_ordering=ORDERING_POLICY, same_day_repeat_policy='retain all; previous fitted observation is the next source',
                    qc_thresholds=QC_THRESHOLDS, qc_flags_are_exclusion_rules=False,
                    scan_table_sha256=sha256(args.scan_table), code_provenance=code_provenance(),
                    environment=dict(torch=str(torch.__version__), numpy=np.__version__, open3d=o3d.__version__,
                                     cuda=torch.version.cuda, pytorch3d=importlib.metadata.version('pytorch3d'),
                                     scipy=importlib.metadata.version('scipy'), python=sys.version),
                    template_sha256=sha256(template_path), face_array_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
                    paper_supplements=dict(displacement_reduction='RMS Euclidean distance',
                        normals='PyTorch3D area-weighted unit vertex normals',
                        reference='initial mesh of each scan', edge=args.edge_mode,
                        graph=args.update_mode + '; detached between optimization steps',
                        surface_sample_count='target boundary point count',
                        iterations='5000 from cited Park mesh method; explicit CLI override for checks',
                        optimizer='AdamW lr=0.0005, StepLR every 1000 gamma=0.5 from cited/public mesh method'),
                    scans=[])
    start_index = 0
    cached_directory = args.resume_from.resolve() if args.resume_from is not None else output if args.resume and (output/'manifest.json').exists() else None
    if args.resume and cached_directory is None and any(output.iterdir()):
        raise ValueError('nonempty output has no verified manifest; inspect interrupted first scan')
    if cached_directory is not None:
        vertices, start_index = reuse_prefix(cached_directory, output, manifest, scans, vertices, faces)
        atomic_manifest(output/'manifest.json', manifest)
        print(f'Reused {start_index} verified scan(s)', flush=True)
    else:
        # An empty prefix lets an interruption during the first fit be resumed.
        atomic_manifest(output/'manifest.json', manifest)
    for index in range(start_index, len(scans)):
        row = scans[index]
        uid = row['uid']
        path = Path(row['parcellation_path'])
        print(f"SCAN {index + 1}/{len(scans)} UID={uid} {row['day']}", flush=True)
        o3d.utility.random.seed(args.seed+index)
        source_hash = sha256(path)
        image = nib.load(path)
        mask = np.asarray(image.dataobj) == 75
        geometry = boundary_geometry(mask, image.affine)
        mask_volume = float(mask.sum()*abs(np.linalg.det(image.affine[:3, :3])))
        mask_points = np.argwhere(mask)
        _, components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
        # Spatial units stay unknown in the source header. The adopted mm
        # interpretation is recorded above, rather than editing the NIfTI.
        target, transform, icp = rigid_target_to_source(
            geometry['target_centered_header_world_units'], vertices,
        )
        source = vertices.copy()
        before = surface_metrics(source, faces, target)
        vertices, elapsed = reconstruct_scan(source, faces, target, args.iterations,
                                            args.device, args.seed + index, output / f'{uid}_loss.csv',
                                            args.update_mode, args.edge_mode)
        after = surface_metrics(vertices, faces, target)
        overlap = mask_overlap(vertices, faces, geometry, transform, mask)
        record = dict(uid=uid, day=row['day'], age=float(row['age']), diagnosis=row['class'],
                      cohort=row.get('cohort', ''), scan_key=row.get('scan_key', uid), split=row.get('split', ''),
                      source_uid=scans[index - 1]['uid'] if index else 'template',
                      initialization=args.template_initialization if index==0 else 'previous fitted mesh, no additional scaling',
                      initialization_factor=initialization_info['factor'] if index==0 else 1.,
                      source_path=str(path), source_sha256=source_hash,
                      source_spatial_units=image.header.get_xyzt_units()[0],
                      mask_volume_mm3=mask_volume, mask_components_26=int(components),
                      mask_touches_image_edge=bool(np.any(mask_points.min(0)==0) or np.any(mask_points.max(0)==np.array(mask.shape)-1)),
                      mask_bbox_voxel=[mask_points.min(0).tolist(), mask_points.max(0).tolist()],
                      volume_bias_percent=float(100*(after['volume_mm3']/mask_volume-1)),
                      boundary_points=len(target), before=before, after=after,
                      overlap=overlap, icp=icp, elapsed_seconds=elapsed,
                      status='candidate_requires_quality_review')
        record['qc_flags'] = review_flags(record)
        record['previous_scan_qc_flags'] = manifest['scans'][-1].get('qc_flags', []) if index else []
        if sha256(path) != source_hash:
            raise RuntimeError('source image changed during fitting')
        with (output/f'{uid}.npz.tmp').open('wb') as file:
            np.savez_compressed(file, vertices=vertices, faces=faces,
                            source_vertices=source, target=target, icp_matrix=transform,
                            centroid_header_world=geometry['centroid_header_world'], affine=geometry['affine'])
        os.replace(output/f'{uid}.npz.tmp', output/f'{uid}.npz')
        write_obj(output/f'{uid}.obj.tmp', vertices, faces)
        os.replace(output/f'{uid}.obj.tmp', output/f'{uid}.obj')
        record['artifact_sha256'] = {uid+suffix: sha256(output/(uid+suffix)) for suffix in ('.npz', '.obj', '_loss.csv')}
        manifest['scans'].append(record)
        atomic_manifest(output/'manifest.json', manifest)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if on_scan_done is not None:
            on_scan_done(record)


if __name__ == '__main__':
    main()
