"""Measure candidate-mesh validity and loss pressure without changing meshes.

A contraction direction is only a diagnostic derivative, not a fitting rule.
No subject exclusion, scale correction, or new training loss is adopted here.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import torch
import trimesh
from pytorch3d.structures import Meshes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh_fit import fitting_loss, WEIGHTS


def inspect_scan(directory, record, device):
    with np.load(directory / f"{record['uid']}.npz") as data:
        vertices, faces = data['vertices'], data['faces']
        source, target = data['source_vertices'], data['target']
    geometry = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    intersections = np.asarray(geometry.get_self_intersecting_triangles())
    v = torch.tensor(vertices, dtype=torch.float32, device=device, requires_grad=True)
    f = torch.tensor(faces, dtype=torch.int64, device=device)
    reference = torch.tensor(source, dtype=torch.float32, device=device)
    reference_mesh = Meshes([reference], [f])
    mesh = Meshes([v], [f])
    p = torch.tensor(target, dtype=torch.float32, device=device)
    # Verify the differentiable volume against the saved-coordinate mesh volume.
    triangles = v[f]
    volume = (triangles[:, 0] * torch.cross(triangles[:, 1], triangles[:, 2], dim=-1)).sum() / 6
    np.testing.assert_allclose(float(volume.detach().cpu()),
                               trimesh.Trimesh(vertices, faces, process=False).volume,
                               rtol=1e-5, atol=.01)
    # Translation does not change volume. Remove the vertex centroid before
    # probing a uniform infinitesimal contraction around that centroid.
    contraction = -(v.detach() - v.detach().mean(0))
    derivative_samples = {key: [] for key in WEIGHTS}
    value_samples = {key: [] for key in WEIGHTS}
    for seed in (71, 72, 73):
        torch.manual_seed(seed)
        current = Meshes([v], [f])
        _, terms = fitting_loss(current, p, reference,
                                reference_mesh.verts_normals_packed(), 'public')
        for key, weight in WEIGHTS.items():
            gradient, = torch.autograd.grad(weight * terms[key], v, retain_graph=True)
            derivative_samples[key].append(float((gradient * contraction).sum().detach().cpu()))
            value_samples[key].append(float((weight * terms[key]).detach().cpu()))
    with (directory / f"{record['uid']}_loss.csv").open() as file:
        history = list(csv.DictReader(file))
    return dict(
        uid=record['uid'],
        source_volume=float(trimesh.Trimesh(source, faces, process=False).volume),
        mesh_volume=float(volume.detach().cpu()),
        self_intersecting_triangle_pairs=len(intersections),
        triangles_in_self_intersections=len(np.unique(intersections)) if intersections.size else 0,
        minimum_triangle_area=float(trimesh.Trimesh(vertices, faces, process=False).area_faces.min()),
        weighted_loss={key: float(np.mean(values)) for key, values in value_samples.items()},
        contraction_derivative={key: float(np.mean(values)) for key, values in derivative_samples.items()},
        derivative_interpretation='positive opposes uniform contraction; negative favors it',
        last_1000_total_loss_change=float(history[-1]['total']) - float(history[-1001]['total'])
            if len(history) > 1000 else None,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    results = [inspect_scan(args.directory, record, args.device) for record in manifest['scans']]
    summary = dict(scans=results, diagnostic_only=True, mesh_vertices_modified=False,
                   contraction_is_not_an_adopted_size_correction=True,
                   note='Self-intersections are distinct from topological watertightness. '
                        'Directional loss derivatives alone do not identify the full fitting cause.')
    (args.directory / 'bias_diagnostics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
