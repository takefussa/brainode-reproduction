"""Verify every train-only pseudo label and Eq.4 shape with independent solves."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cognition import RK4ShapeIntegrator
from lab_shapes import ShapePCA
from model import ODEFuncWithAttention


def main():
    report = PROJECT/'artifacts/training_runs/adni1_continuous_v1'
    config = json.loads((report/'config.json').read_text())
    root = Path(config['output'])/'pseudo'
    manifest = json.loads((root/'manifest.json').read_text())
    with np.load(root/'samples.npz', allow_pickle=False) as archive:
        samples = {k: archive[k].copy() for k in archive.files}
    with np.load(Path(config['data_root'])/'observations.npz', allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    pca = ShapePCA.load(Path(config['data_root'])/'pca.npz')
    torch.set_num_threads(1)
    model = ODEFuncWithAttention(150, 1)
    model.load_state_dict(torch.load(config['initial_checkpoint'], map_location='cpu', weights_only=False)['model']); model.eval()
    integrate = RK4ShapeIntegrator(model)
    difference = []
    with torch.inference_mode():
        for i, record in enumerate(manifest['records']):
            source, target = record['source_index'], record['target_index']
            if (data['splits'][source] != 'train' or data['splits'][target] != 'train' or
                    data['pids'][source] != data['pids'][target] or data['conditions'][source] != 0 or data['conditions'][target] != 1):
                raise ValueError('invalid training endpoint')
            start, final, current = float(data['times'][source]), float(data['times'][target]), record['time']
            np.testing.assert_allclose(record['c'], (current-start)/(final-start), atol=1e-12, rtol=0)
            forward = integrate(data['coefficients'][source], start, current, 0.)
            backward = integrate(data['coefficients'][target], final, current, 1.)
            expected = ((forward+backward)/2).numpy()
            reconstructed = pca.inverse_transform(expected[None]).reshape(-1, 3)
            error = float(np.linalg.norm(reconstructed-samples['vertices'][i], axis=1).max())
            if error > 1e-4:
                raise ValueError('saved Eq.4 sample differs from independent CPU RK4 solve')
            difference.append(error)
            if hashlib.sha256(Path(record['file']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('pseudo occupancy changed')
    summary = dict(passed=True, samples=len(difference), patients=manifest['subjects'], pairs=manifest['pairs'],
        max_vertex_difference_cpu_vs_gpu_assumed_mm=max(difference), tolerance_max_vertex_assumed_mm=1e-4,
        train_only_endpoints_verified=True, relative_time_c_verified=True, observed_data_excluded=0,
        independent_check='two explicit CPU RK4 solves averaged by 1/2; compare GPU-generated saved shapes',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (report/'pseudo/verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for pair in range(manifest['pairs']):
        records = [r for r in manifest['records'] if r['pair_index'] == pair]
        label = f"{records[0]['pid']} CN:{records[0]['source_uid']}"
        axes[0].plot([r['c'] for r in records], [r['signed_mesh_volume_mm3'] for r in records], marker='.', label=label)
        axes[1].plot([r['c'] for r in records], [r['affected_area_percent'] for r in records], marker='.', label=label)
    axes[0].set(xlabel='Synthetic c', ylabel='Signed PCA mesh volume (assumed mm3)')
    axes[1].set(xlabel='Synthetic c', ylabel='Intersection-affected surface area (%)')
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.savefig(report/'pseudo/geometry.png', dpi=180); fig.savefig(report/'pseudo/geometry.pdf'); plt.close(fig)
    titles = ['CN endpoint (PCA)', 'Pseudo c=0.1', 'Pseudo c=0.5', 'Pseudo c=0.9', 'AD endpoint (PCA)']
    viewer = make_subplots(rows=1, cols=5, specs=[[{'type': 'scene'}]*5], subplot_titles=titles)
    faces = data['faces']; all_shapes = []
    for pair in range(manifest['pairs']):
        selected = [r for r in manifest['records'] if r['pair_index'] == pair]
        first = selected[0]
        shapes = [pca.inverse_transform(data['coefficients'][first['source_index']:first['source_index']+1]).reshape(-1, 3)]
        shapes.extend(samples['vertices'][next(r['index'] for r in selected if np.isclose(r['c'], c))] for c in [.1, .5, .9])
        shapes.append(pca.inverse_transform(data['coefficients'][first['target_index']:first['target_index']+1]).reshape(-1, 3))
        all_shapes.extend(shapes)
        for column, (v, color) in enumerate(zip(shapes, ['#87939b', '#b0cd86', '#dda345', '#de776c', '#2697ce']), 1):
            viewer.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, visible=pair == 0, name=titles[column-1]), row=1, col=column)
    buttons = []
    for pair in range(manifest['pairs']):
        r = next(r for r in manifest['records'] if r['pair_index'] == pair)
        title = f"Train-only Eq.4: {r['pid']}, CN {r['source_uid']} to AD {r['target_uid']}; canonical frame"
        buttons.append(dict(label=f"{r['pid']} CN {r['source_uid']}", method='update',
                            args=[dict(visible=[i//5 == pair for i in range(5*manifest['pairs'])]), dict(title=title)]))
    union = np.concatenate(all_shapes); lower, upper = union.min(0)-1, union.max(0)+1
    viewer.update_scenes(aspectmode='data', xaxis=dict(range=[lower[0], upper[0]]),
                         yaxis=dict(range=[lower[1], upper[1]]), zaxis=dict(range=[lower[2], upper[2]]))
    viewer.update_layout(height=650, title=buttons[0]['args'][1]['title'], updatemenus=[dict(buttons=buttons, x=0, y=1.2)])
    viewer.write_html(report/'pseudo/preview.html', include_plotlyjs=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
