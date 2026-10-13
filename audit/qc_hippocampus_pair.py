"""Read-only spatial QA of two ADNI1 scans; outputs only under this project.

The selected pair is an inspection example, not cohort eligibility. Only two
unified MRI files are read to resolve the mask's frame. Display projection uses
header affine alone and is not an accepted native-to-TRSAA registration.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mesh_input import boundary_geometry

SOURCE = Path('/data2/radiology_datas/clean3/ADNI1')
OUTPUT = PROJECT / 'audit' / 'hippocampus_pair_qc'
UIDS = ('34698', '162792')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def header(image):
    return {
        'shape': list(image.shape), 'affine': image.affine.tolist(),
        'spacing_numeric': [float(x) for x in image.header.get_zooms()],
        'spatial_units': image.header.get_xyzt_units()[0],
        'orientation': list(nib.aff2axcodes(image.affine)),
        'qform_code': int(image.header['qform_code']),
        'sform_code': int(image.header['sform_code']),
    }


def overlay_views(axes, image, mask, mask_affine, centroid_world, title):
    volume = np.asarray(image.dataobj)
    center = nib.affines.apply_affine(np.linalg.inv(image.affine), centroid_world)
    center = np.clip(np.rint(center).astype(int), 0, np.array(image.shape) - 1)
    display_to_mask = np.linalg.inv(mask_affine) @ image.affine
    bright = volume[volume > 0]
    vmax = float(np.percentile(bright, 99.5)) if len(bright) else 1
    for axis, (dimension, name) in zip(axes, [(2, 'axial'), (1, 'coronal'), (0, 'sagittal')]):
        index = int(center[dimension])
        remaining = [d for d in range(3) if d != dimension]
        plane = np.take(volume, index, axis=dimension)
        coordinates = np.empty((3, plane.size), dtype=np.float64)
        coordinates[dimension] = index
        grid = np.indices(plane.shape).reshape(2, -1)
        coordinates[remaining] = grid
        native = display_to_mask[:3, :3] @ coordinates + display_to_mask[:3, 3, None]
        projected = ndimage.map_coordinates(
            mask.astype(np.uint8), native, order=0, mode='constant', cval=0,
        ).reshape(plane.shape)
        axis.imshow(plane.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        if projected.any() and not projected.all():
            axis.contour(projected.T, levels=[0.5], colors=['#ff5964'], linewidths=1)
        axis.set_title(f'{title}\n{name}, index {index}', fontsize=10)
        axis.set_xlabel(f'voxel axis {remaining[0]}')
        axis.set_ylabel(f'voxel axis {remaining[1]}')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (PROJECT / 'audit' / 'adni1_scans.csv').open(newline='') as file:
        rows = {row['uid']: row for row in csv.DictReader(file)}
    template_path = PROJECT / 'dataset' / 'Shapes' / 'hippocampus_template_mesh.obj'
    # process=False preserves the supplied vertex/face order.
    template = trimesh.load_mesh(template_path, process=False)
    assert template.vertices.shape == (1490, 3) and template.faces.shape == (2976, 3)
    records = []
    for uid in UIDS:
        row = rows[uid]
        assert row['pid'] == '041_S_0898'
        paths = {
            'parcellation': Path(row['parcellation_path']),
            'trsaa': Path(row['path_full_mni']),
            'rigid': Path(row['path_full']),
            'unified': SOURCE / 'unified' / f'{uid}.nii',
        }
        images = {key: nib.load(path) for key, path in paths.items()}
        segmentation = np.asarray(images['parcellation'].dataobj)
        mask = segmentation == 75
        assert mask.any()
        voxels = np.argwhere(mask)
        geometry = boundary_geometry(mask, images['parcellation'].affine)
        world = nib.affines.apply_affine(images['parcellation'].affine, voxels)
        _, components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
        record = {
            'pid': row['pid'], 'uid': uid, 'day': row['day'],
            'age': float(row['age']), 'diagnosis': row['class'], 'label': 75,
            'paths': {k: str(v) for k, v in paths.items()},
            'source_sha256': {k: sha256(v) for k, v in paths.items()},
            'headers': {k: header(v) for k, v in images.items()},
            'mask_voxels': int(mask.sum()), 'connected_components_26': int(components),
            'mask_bbox_voxel': [voxels.min(0).tolist(), voxels.max(0).tolist()],
            'mask_centroid_header_world': world.mean(0).tolist(),
            'mask_extent_header_world_numeric': np.ptp(world, axis=0).tolist(),
            'volume_numeric_if_header_units_are_mm': float(mask.sum() * abs(np.linalg.det(images['parcellation'].affine[:3, :3]))),
            'boundary_points': len(geometry['boundary_voxel']),
            'centered_voxel_vs_header_world_max_abs_diff': float(np.abs(
                geometry['target_centered_voxel_units'] - geometry['target_centered_header_world_units']
            ).max()),
            'unit_confirmation': 'unknown in all four headers; mm is not yet provenance-confirmed',
            'projection_is_registration': False,
        }
        for key in ['unified', 'rigid', 'trsaa']:
            image = images[key]
            indices = np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine), world)).astype(int)
            inside = np.all((indices >= 0) & (indices < np.array(image.shape)), axis=1)
            intensities = np.asarray(image.dataobj)[tuple(indices[inside].T)]
            record[f'parcellation_vs_{key}'] = {
                'same_shape': images['parcellation'].shape == image.shape,
                'same_affine': bool(np.allclose(images['parcellation'].affine, image.affine, atol=1e-5, rtol=0)),
                'affine_projected_inside_fraction': float(inside.mean()),
                'affine_projected_zero_intensity_fraction': float((intensities == 0).mean()),
            }
        # These are boundary targets before ICP, not deformed meshes.
        np.savez_compressed(
            OUTPUT / f'{uid}_boundary.npz', **geometry,
            label=np.array(75), spatial_units=np.array(record['headers']['parcellation']['spatial_units']),
            template_vertices=template.vertices, template_faces=template.faces,
        )
        fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
        overlay_views(axes[0], images['unified'], mask, images['parcellation'].affine, world.mean(0), 'unified: matching header frame')
        overlay_views(axes[1], images['trsaa'], mask, images['parcellation'].affine, world.mean(0), 'TRSAA: affine-only diagnostic projection')
        fig.suptitle(f"{row['pid']} | UID {uid} | {row['day']} | {row['class']}\nRed: label 75. Bottom projection has no native-to-TRSAA registration.", fontsize=12)
        fig.savefig(OUTPUT / f'{uid}_space_qc.png', dpi=140)
        plt.close(fig)
        records.append(record)
        print(uid, 'mask voxels', record['mask_voxels'], 'boundary points', record['boundary_points'], 'connected components', components)
    summary = {
        'scope': 'two-scan QC example; no training cohort selection or fitted mesh',
        'template': {
            'path': str(template_path), 'sha256': sha256(template_path),
            'vertices': len(template.vertices), 'faces': len(template.faces),
            'watertight': bool(template.is_watertight), 'euler_number': int(template.euler_number),
            'extent_numeric': np.ptp(template.vertices, axis=0).tolist(),
        },
        'records': records,
        'next_input_contract': {
            'source_first_timepoint': 'provided 1490-vertex template with fixed faces/index',
            'source_later_timepoint': 'previous fitted mesh for every retained chronological observation',
            'target': 'label75 boundary PCD, centered and then rigid ICP to source',
            'pending': ['confirm spatial mm interpretation', 'decide native-mask vs TRSAA-aligned-mask geometry', 'resolve paper fitting loss details before optimization'],
        },
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
