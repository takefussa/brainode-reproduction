"""Independently validate and present fixed-final continuous-c results."""

import csv
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluation import shape_errors
from lab_shapes import ShapePCA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def patient_values(records, keys):
    groups = {}
    for record in records:
        groups.setdefault(record['pid'], []).append(record)
    return {pid: np.array([[float(r[k]) for k in keys] for r in group]).mean(0)
            for pid, group in groups.items()}


def paired_summary(records, keys):
    patients = patient_values(records, keys)
    values = np.stack(list(patients.values()))
    # Uncertainty resamples patients, preserving within-patient target dependence.
    generator = np.random.default_rng(42)
    boot = values[generator.integers(0, len(values), size=(10000, len(values)))].mean(1)
    return dict(patients=len(patients), targets=len(records),
        patient_macro_mean=dict(zip(keys, values.mean(0).tolist())),
        differences_vs_initial={key: dict(mean=float((values[:, i]-values[:, 0]).mean()),
            patient_bootstrap_95_percent_interval=np.quantile(boot[:, i]-boot[:, 0], [.025, .975]).tolist(),
            improved_patients=int(np.sum(values[:, i] < values[:, 0]))) for i, key in enumerate(keys) if i},
        primary='patient macro mean corresponding vertex Euclidean distance to original observed mesh')


def main():
    report = PROJECT/'artifacts/training_runs/adni1_continuous_v1'
    config = json.loads((report/'deterministic_pool_config.json').read_text())
    root = Path(config['output'])
    for stage in ('pseudo', 'cnn', 'ode', 'evaluate'):
        status = json.loads((report/stage/'progress.json').read_text())
        if status['status'] != 'completed':
            raise ValueError('stage incomplete: '+stage)
        if stage in ('cnn', 'ode') and digest(root/stage/'latest.pt') != status['checkpoint_sha256']:
            raise ValueError('final checkpoint changed: '+stage)
    evaluation = json.loads((report/'evaluate/evaluation.json').read_text())
    if digest(root/'evaluate/future_predictions.npz') != evaluation['prediction_file_sha256']:
        raise ValueError('prediction archive changed')
    with np.load(Path(config['data_root'])/'observations.npz', allow_pickle=False) as file:
        data = {k: file[k].copy() for k in file.files}
    with np.load(root/'evaluate/future_predictions.npz', allow_pickle=False) as file:
        predictions = {k: file[k].copy() for k in file.files}
    np.testing.assert_array_equal(predictions['faces'], data['faces'])
    indices = {str(uid): i for i, uid in enumerate(data['uids'])}
    pca = ShapePCA.load(Path(config['data_root'])/'pca.npz')
    metrics = rows(report/'evaluate/future_metrics.csv')
    original = rows(PROJECT/'artifacts/training_runs/adni1_observed_v1/future_metrics.csv')
    matched_report = json.loads((report/'evaluate/matched_controls.json').read_text())
    if digest(root/'evaluate/matched_controls.npz') != matched_report['predictions_sha256']:
        raise ValueError('matched control archive changed')
    matched = rows(report/'evaluate/matched_controls.csv')
    if [(r['pid'], r['source_uid'], r['target_uid']) for r in matched] != [(r['pid'], r['source_uid'], r['target_uid']) for r in metrics]:
        raise ValueError('matched controls change evaluation coverage')
    if [(r['pid'], r['source_uid'], r['target_uid']) for r in original] != [(r['pid'], r['source_uid'], r['target_uid']) for r in metrics]:
        raise ValueError('comparison changes future targets')
    if len(metrics) != 175:
        raise ValueError('future coverage changed')
    for i, (record, baseline) in enumerate(zip(metrics, original)):
        source, target = indices[record['source_uid']], indices[record['target_uid']]
        if data['splits'][target] != 'test' or data['splits'][source] != 'test' or data['pids'][target] != record['pid']:
            raise ValueError('patient/split mismatch')
        distance = shape_errors(predictions['vertices'][i], data['vertices'][target])['mean_vertex_euclidean']
        if abs(distance-float(record['prediction_mean_vertex_euclidean'])) > 1e-10:
            raise ValueError('saved prediction metric does not reproduce')
        record['initial_observed_ode_euclidean'] = float(baseline['prediction_mean_vertex_euclidean'])
        smooth = pca.inverse_transform(data['coefficients'][source:source+1]).reshape(-1, 3)
        record['static_pca_source_euclidean'] = shape_errors(smooth, data['vertices'][target])['mean_vertex_euclidean']
        for key in matched_report['controls']:
            record[key] = float(matched[i][key])
    last = [r for r in metrics if float(r['target_time']) == max(float(s['target_time']) for s in metrics if s['pid'] == r['pid'])]
    if len(last) != 62:
        raise ValueError('latest-day repeat coverage changed')
    keys = ['initial_observed_ode_euclidean', 'augmented_fixed_c_euclidean',
            'prediction_mean_vertex_euclidean', 'static_source_mean_vertex_euclidean', 'static_pca_source_euclidean']
    result = dict(all=paired_summary(metrics, keys), latest=paired_summary(last, keys),
        cognition=json.loads((report/'cnn/evaluation.json').read_text()),
        pseudo=json.loads((report/'pseudo/progress.json').read_text()), geometry=evaluation,
        units='assumed mm; original spatial header units unspecified',
        testing='patient split fixed; final epochs and Algorithm 2 steps fixed before held-out evaluation')
    matched_keys = matched_report['controls'] + ['prediction_mean_vertex_euclidean']
    result['matched_condition_controls'] = dict(all=paired_summary(metrics, matched_keys), latest=paired_summary(last, matched_keys))
    result['algorithm2_reference_verification'] = json.loads((report/'evaluate/reference_verification.json').read_text())
    write = lambda path, value: path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write(report/'RESULTS.json', result)
    with (report/'paired_comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    cnn = rows(report/'cnn/epochs.csv'); ode = rows(report/'ode/epochs.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axes[0].plot([int(r['epoch']) for r in cnn], [float(r['train_soft_bce']) for r in cnn], label='All training shapes')
    axes[0].plot([int(r['epoch']) for r in cnn], [float(r['pseudo_soft_bce']) for r in cnn], label='Pseudo only', alpha=.7)
    axes[0].set(xlabel='CNN epoch', ylabel='Soft-label BCE'); axes[0].legend(fontsize=8)
    axes[1].plot([int(r['epoch']) for r in ode], [float(r['train_patient_mean_squared_pca_l2']) for r in ode])
    axes[1].set(xlabel='Additional ODE epoch', ylabel='Patient mean squared PCA L2')
    patients = patient_values(last, keys)
    values = np.stack(list(patients.values()))
    axes[2].scatter(values[:, 0], values[:, 2], s=18, alpha=.75)
    limit = values[:, [0, 2]].max()*1.05
    axes[2].plot([0, limit], [0, limit], color='gray', lw=1)
    axes[2].set(xlim=(0, limit), ylim=(0, limit), xlabel='Initial observed ODE error', ylabel='Algorithm 2 error')
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(report/'results.png', dpi=180); fig.savefig(report/'results.pdf'); plt.close(fig)
    # Reuse the original baseline's displayed patient/target for direct visual comparison.
    preview = json.loads((PROJECT/'artifacts/training_runs/adni1_observed_v1/RESULTS.json').read_text())['preview']
    index = next(i for i, r in enumerate(metrics) if r['pid'] == preview['pid'] and r['target_uid'] == preview['target_uid'])
    r = metrics[index]; source, target = indices[r['source_uid']], indices[r['target_uid']]
    with np.load('/data5/take/brainode/experiments/adni1_observed_v1/future_predictions.npz', allow_pickle=False) as archive:
        initial_prediction = archive['vertices'][index]
    viewer = make_subplots(rows=1, cols=4, specs=[[{'type': 'scene'}]*4],
        subplot_titles=['Initial observed mesh', 'Future observed mesh', 'Initial observed ODE', 'Continuous c: Algorithm 2'])
    shapes = [data['vertices'][source], data['vertices'][target], initial_prediction, predictions['vertices'][index]]
    for column, (v, color) in enumerate(zip(shapes, ['#87939b', '#2697ce', '#e8a838', '#58b986']), 1):
        faces = data['faces']
        viewer.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=faces[:, 0], j=faces[:, 1],
                                  k=faces[:, 2], color=color, name=str(column)), row=1, col=column)
    union = np.concatenate(shapes)
    lower, upper = union.min(0)-1, union.max(0)+1
    viewer.update_scenes(aspectmode='data', xaxis=dict(range=[lower[0], upper[0]]),
                         yaxis=dict(range=[lower[1], upper[1]]), zaxis=dict(range=[lower[2], upper[2]]))
    viewer.update_layout(title=f"{r['pid']} target {r['target_uid']}: Algorithm 2 error {float(r['prediction_mean_vertex_euclidean']):.3f} assumed mm; canonical frame", height=650)
    viewer.write_html(report/'prediction_preview.html', include_plotlyjs=True)
    (report/'preview').mkdir(exist_ok=True)
    import trimesh
    for name, shape in zip(['source', 'target_observed', 'initial_ode', 'algorithm2'], shapes):
        trimesh.Trimesh(vertices=shape, faces=data['faces'], process=False).export(report/'preview'/f'{name}.ply')
    means = result['latest']['patient_macro_mean']
    test = result['cognition']['test']
    matched_means = result['matched_condition_controls']['latest']['patient_macro_mean']
    lines = ['# ADNI1 continuous-c results', '',
        'All three stages are complete: 36 train-only Eq.4 pseudo shapes, a voxel-input cognition estimator, and augmented BrainODE plus Algorithm 2 forecasting.', '',
        'CNN: 30 fixed epochs. BrainODE: 100 initial observed epochs followed by 20 augmented epochs. Algorithm 2: 10 fixed steps. Unpublished optimization, mixing and sampling settings are implementation supplements.', '',
        '| Method | Latest-day patient macro vertex error (assumed mm) |', '|---|---:|',
        f"| Initial observed-only BrainODE | {means[keys[0]]:.4f} |",
        f"| Augmented BrainODE, observed c fixed, ten steps | {matched_means[matched_keys[1]]:.4f} |",
        f"| Augmented BrainODE, initial CNN c fixed, ten steps | {matched_means[matched_keys[2]]:.4f} |",
        f"| Augmented BrainODE, Algorithm 2 estimated c | {means[keys[2]]:.4f} |",
        f"| Static original source mesh | {means[keys[3]]:.4f} |", '',
        'The comparison uses the same 60 held-out temporal patients and all 62 latest-day scans, including repeats. All 175 later targets are also reported. Metrics are against original meshes, not only PCA targets. Patient bootstrap intervals are descriptive uncertainty estimates; the test set never selected checkpoints or step counts.', '',
        'Matched ten-step controls use observed source c or the initial CNN estimate held constant, isolating condition updates from solver segmentation. Augmented versus original training combines additional optimization and pseudo supervision; this comparison cannot establish a pseudo-only benefit.', '',
        f"CNN held-out observed-label accuracy: {test['accuracy']:.3f}; balanced accuracy: {test['balanced_accuracy']:.3f}; ROC AUC: {test['roc_auc']:.3f} across 255 scans in 79 patients. Fractional c has no measured clinical ground truth. Two train patients supply all synthetic conversion labels; there are no held-out converters.", '',
        f"Self-intersection remains in {evaluation['predictions_with_self_intersection']}/175 outputs; affected area maximum {evaluation['max_predicted_affected_area_percent']:.4f}%. All 1,750 intermediate inputs and 175 final outputs were checked on the fixed grid; three/seven-ray discrepancies: {evaluation['ray_vote_disagreements']}. Signed-volume checks do not prove correct anatomical interiors.", '',
        '[Full numbers](RESULTS.json), [paired scan errors](paired_comparison.csv), [learning and paired results](results.png), [interactive 3D comparison](prediction_preview.html). PLY meshes under `preview/` open in Slicer in the shared canonical frame, not registered native MRI coordinates.', '',
        'Reproduction provenance: original `config.json`/`code_snapshot/` for pseudo and ODE, `deterministic_pool_config.json`/`code_snapshot_deterministic_pool/` for CNN and evaluation. The initial CNN pooling failure before any optimizer update is retained separately. Original source data and prior completed baseline artifacts remain unchanged.', '']
    (report/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(latest=result['latest'], cognition=test, intersections=evaluation['predictions_with_self_intersection']), indent=2))


if __name__ == '__main__':
    main()
