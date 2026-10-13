"""Render saved observed meshes; no fitting or image preprocessing."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from audit.render_mesh_qc import interactive_series


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--subject-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.subject_directory.resolve(), args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error('light visualizations must remain under the home project')
    manifest = json.loads((directory/'manifest.json').read_text())
    if not manifest['scans']:
        parser.error('at least one completed scan is required')
    output.mkdir(parents=True, exist_ok=True)
    loaded, metrics = [], []
    for record in manifest['scans']:
        for name, digest in record['artifact_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError('saved candidate artifact changed')
        with np.load(directory/f"{record['uid']}.npz") as arrays:
            data = {key: arrays[key].copy() for key in arrays.files}
        loaded.append((record, data))
        with (directory/f"{record['uid']}_loss.csv").open() as file:
            history = list(csv.DictReader(file))
        metrics.append(dict(uid=record['uid'], day=record['day'], diagnosis=record['diagnosis'],
            volume_bias_percent=record['volume_bias_percent'], dice=record['overlap']['dice'],
            self_intersections=record['after']['self_intersecting_triangle_pairs'],
            qc_flags=record['qc_flags'], first_logged_loss=float(history[0]['total']),
            last_logged_loss=float(history[-1]['total']),
            loss_iteration_count=int(history[-1]['iteration']),
            mask_volume=record['mask_volume_mm3'], mesh_volume=record['after']['volume_mm3']))
    points = np.vstack([data[key] for _, data in loaded for key in ('source_vertices','vertices','target')])
    lower, upper = points.min(0)-1, points.max(0)+1
    figure = plt.figure(figsize=(12, 8), layout='constrained')
    for index, (record, data) in enumerate(loaded):
        if len(loaded) > 4:
            parser.error('preview supports up to four saved scans; use the full interactive viewer for longer series')
        ax = figure.add_subplot(2, 2, index+1, projection='3d')
        vertices, faces, target = data['vertices'], data['faces'], data['target']
        ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor='#2697ce',
            edgecolor='#237ea8', linewidth=.08, alpha=.70))
        ax.scatter(*target.T, s=.8, color='#d64536', alpha=.30, depthshade=False)
        for dimension, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(lower[dimension], upper[dimension])
        ax.set_box_aspect(upper-lower)
        ax.view_init(elev=22, azim=-130)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title(f"{record['day']} / {record['diagnosis']} / UID {record['uid']}\n"
            f"Volume bias {record['volume_bias_percent']:+.2f}% | Dice {record['overlap']['dice']:.3f} | "
            f"Intersections {record['after']['self_intersecting_triangle_pairs']}", fontsize=10)
    figure.suptitle(f"ADNI1 observed hippocampal meshes: {manifest['subject']}\n"
        'Blue: fitted mesh. Red: original parcellation boundary.\n'
        'Same display axes across visits; these are observed reconstructions, not future predictions.', fontsize=12)
    figure.savefig(output/'mesh_preview.png', dpi=160)
    figure.savefig(output/'mesh_preview.pdf')
    plt.close(figure)
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.4*len(loaded), 3.2), layout='constrained', squeeze=False)
    for index, (record, _) in enumerate(loaded):
        with (directory/f"{record['uid']}_loss.csv").open() as file:
            history = list(csv.DictReader(file))
        ax = axes[0,index]
        ax.plot([int(r['iteration']) for r in history], [float(r['total']) for r in history], color='#237fa8')
        ax.set_title(record['day']+' / '+record['diagnosis'], fontsize=10)
        ax.set_xlabel('Optimizer update'); ax.set_ylabel('Total fitting loss'); ax.grid(alpha=.2)
    fig.suptitle('Per-scan optimization: weighted total loss before each update', fontsize=11)
    fig.savefig(output/'optimization_loss.png', dpi=160)
    plt.close(fig)
    interactive_series(directory, manifest, output/'mesh_3d.html')
    (output/'metrics.json').write_text(json.dumps(dict(subject=manifest['subject'],
        source_directory=str(directory), purpose='completed observed mesh preview; no scientific cohort selection',
        predicted_shapes=False, iterations=manifest['iterations'],
        spatial_units='1 numeric unit assumed to be 1 mm', scans=metrics), indent=2)+'\n')
    lines=['# ADNI1の完成済みmesh表示','',
        f"対象: {manifest['subject']}。{len(metrics)}時点の観測meshを表示する。",'',
        '青が生成mesh、赤が元parcellationの境界点。各撮像のrigid ICP後の座標で比較し、同じ表示範囲を使う。未来形状の予測ではない。元parcellationや保存meshを変更していない。', '',
        '[比較画像](mesh_preview.png)、[回転と日付切替ができる3D](mesh_3d.html)、[最適化loss](optimization_loss.png)、[全数値](metrics.json)。','',
        'この患者の画像は表示例であり、全対象の品質を代表する統計や学習採用を意味しない。', '',
        '| 日付 | 診断 | 体積差 | Dice | 自己交差 | 最初→最後のlogged total loss |',
        '|---|---|---:|---:|---:|---:|']
    for record in metrics:
        lines.append(f"| {record['day']} | {record['diagnosis']} | {record['volume_bias_percent']:+.2f}% | "
            f"{record['dice']:.3f} | {record['self_intersections']} | "
            f"{record['first_logged_loss']:.3f} → {record['last_logged_loss']:.3f} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
