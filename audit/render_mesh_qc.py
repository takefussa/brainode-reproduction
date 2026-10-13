"""Compare candidate fitted surfaces with their original boundary targets."""

import csv
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def interactive_series(directory, manifest, output=None):
    """Standalone scientific 3D artifact; no image/mesh preprocessing."""
    import plotly.graph_objects as go
    frames, extent_points = [], []
    for record in manifest['scans']:
        data = np.load(directory / f"{record['uid']}.npz")
        vertices, faces, target = data['vertices'], data['faces'], data['target']
        extent_points.extend([vertices, target])
        traces = [
            go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                      i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='#409fda',
                      opacity=.85, name='Fitted mesh', showlegend=True),
            go.Scatter3d(x=target[:, 0], y=target[:, 1], z=target[:, 2],
                         mode='markers', marker=dict(size=1, color='#da4938', opacity=.25),
                         name='Observed mask boundary'),
        ]
        title = (f"Observed hippocampal mesh: {manifest['subject']} / {record['uid']}"
                 f"<br>{record['day']}, age {record['age']:.4f}, {record['diagnosis']}; "
                 f"Dice {record['overlap']['dice']:.3f}<br>Coordinates assume 1 numeric unit = 1 mm")
        frames.append(go.Frame(name=record['uid'], data=traces, layout=go.Layout(title=title)))
    figure = go.Figure(data=frames[0].data, frames=frames)
    all_points = np.vstack(extent_points)
    lower, upper = all_points.min(0) - 1, all_points.max(0) + 1
    figure.update_layout(
        title=frames[0].layout.title, width=1100, height=780,
        scene=dict(aspectmode='data',
                   xaxis=dict(title='x (assumed mm)', range=[lower[0], upper[0]]),
                   yaxis=dict(title='y (assumed mm)', range=[lower[1], upper[1]]),
                   zaxis=dict(title='z (assumed mm)', range=[lower[2], upper[2]]),
                   camera=dict(projection=dict(type='orthographic'))),
        sliders=[dict(active=0, steps=[dict(label=record['day']+' / '+record['uid'], method='animate',
                      args=[[record['uid']], dict(mode='immediate', frame=dict(duration=0, redraw=True),
                                                 transition=dict(duration=0))])
                                      for record in manifest['scans']])],
    )
    path = Path(output) if output is not None else directory / 'mesh_3d.html'
    figure.write_html(path, include_plotlyjs=True, auto_play=False)
    print(path)


def main(directories, output=None):
    directories = [directory.resolve() for directory in directories]
    entries = []
    for directory in directories:
        manifest = json.loads((directory / 'manifest.json').read_text())
        for record in manifest['scans']:
            entries.append((directory, manifest, record))
    count = len(entries)
    figure = plt.figure(figsize=(5 * count, 9))
    for column, (directory, manifest, record) in enumerate(entries):
        uid = record['uid']
        data = np.load(directory / f'{uid}.npz')
        vertices, faces, target = data['vertices'], data['faces'], data['target']
        axis = figure.add_subplot(2, count, column + 1, projection='3d')
        surface = Poly3DCollection(vertices[faces], alpha=.55, facecolor='#409fda', linewidth=0)
        axis.add_collection3d(surface)
        axis.scatter(*target.T, s=.7, c='#da4938', alpha=.25)
        union = np.vstack([vertices, target])
        center, half = (union.max(0) + union.min(0)) / 2, np.ptp(union, axis=0).max() / 2 * 1.05
        for setter, coordinate in zip([axis.set_xlim, axis.set_ylim, axis.set_zlim], center):
            setter(coordinate - half, coordinate + half)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=20, azim=-60)
        axis.set_xlabel('x (assumed mm)'); axis.set_ylabel('y'); axis.set_zlabel('z')
        mean = record['after']['boundary_to_surface_mean_mm']
        dice = record['overlap']['dice']
        dice_label = f'{dice:.3f}' if dice is not None else 'not computed'
        method = f"{manifest['update_mode']} / {manifest['edge_mode']}; {manifest['iterations']} iterations"
        method += ('\nPublic vertex MSE; normal omitted'
                   if manifest.get('recipe_kind') == 'diagnostic_author_loss'
                   else '\nVertex/normal RMS supplement')
        axis.set_title(f"{uid}  {record['day']}\n{method}\nBoundary distance={mean:.3f} mm; Dice={dice_label}", fontsize=10)
        axis = figure.add_subplot(2, count, count + column + 1)
        with (directory / f'{uid}_loss.csv').open() as file:
            history = list(csv.DictReader(file))
        iterations = [int(row['iteration']) for row in history]
        colors = dict(chamfer='C0', point_mesh='C1', vertex='C2', normal='C3',
                      edge='C4', consistency='C5', laplacian='C6')
        for key, weight in manifest['weights'].items():
            if weight == 0:
                continue
            axis.plot(iterations, [max(float(row[key]) * weight, 1e-8) for row in history],
                      label=key, linewidth=.8, color=colors[key])
        axis.set_yscale('log')
        axis.set_xlabel('Optimization iteration')
        axis.set_ylabel('Weighted loss component')
        axis.grid(alpha=.2)
        if column == 0:
            axis.legend(fontsize=7)
    figure.suptitle(f"{manifest['subject']}: candidate mesh (blue), mask boundary (red)\n"
                   "Iterations and loss definitions are shown for each fit; fixed 1490 vertices / 2976 faces\n"
                   'Unit/reduction supplements are documented; these meshes require quality review.', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .9))
    path = output if output is not None else (directories[0] / 'mesh_qc.png' if len(directories) == 1
            else directories[0].parent / 'comparison_qc.png')
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(path)
    if len(directories) == 1 and len(manifest['scans']) > 1:
        interactive_series(directories[0], manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, help='explicit figure path to preserve earlier comparisons')
    args = parser.parse_args()
    main(args.directories, output=args.output)
