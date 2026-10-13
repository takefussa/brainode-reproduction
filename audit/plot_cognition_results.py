"""Inspect estimated c, patient weighting and geometry strata after fixed training."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def group_metrics(records):
    y = np.array([int(float(r['observed_c'])) for r in records])
    p = np.array([float(r['estimated_c']) for r in records])
    groups = {}
    for row, correct in zip(records, (p >= .5) == y):
        groups.setdefault(row['pid'], []).append(float(correct))
    return dict(scans=len(records), patients=len(groups),
        scan_accuracy=float(np.mean((p >= .5) == y)),
        patient_macro_accuracy=float(np.mean([np.mean(v) for v in groups.values()])),
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        brier=float(np.mean((p-y)**2)), class_counts_CN_AD=np.bincount(y, minlength=2).tolist())


def main():
    report = PROJECT/'artifacts/training_runs/adni1_continuous_v1'
    status = json.loads((report/'cnn/progress.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('fixed final CNN training and evaluation required')
    records = read(report/'cnn/cognition_predictions.csv')
    geometry = {r['uid']: r for r in read(PROJECT/'artifacts/model_data/adni1_pca150_v1/geometry/geometry.csv')}
    test = [r for r in records if r['split'] == 'test']
    intersecting = [r for r in test if int(geometry[r['uid']]['intersection_pairs']) > 0]
    nonintersecting = [r for r in test if int(geometry[r['uid']]['intersection_pairs']) == 0]
    summary = dict(test=group_metrics(test),
        test_pca_self_intersecting=group_metrics(intersecting),
        test_pca_nonintersecting=group_metrics(nonintersecting),
        limitation='Nonrandom geometry strata differ in diagnosis and morphology. Differences are descriptive, not a causal effect of mesh intersections.',
        threshold_fixed=.5, model_selection='fixed final epoch; no test tuning')
    (report/'cnn/strata.json').write_text(json.dumps(summary, indent=2)+'\n')
    pseudo = json.loads((report/'pseudo/manifest.json').read_text())
    pseudo_predictions = {int(r['uid'].split(':')[1]): float(r['estimated_c']) for r in records if r['pseudo'] == 'True'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for y, label, color in [(0, 'Observed CN', '#2697ce'), (1, 'Observed AD', '#df7460')]:
        p = [float(r['estimated_c']) for r in test if float(r['observed_c']) == y]
        axes[0].hist(p, bins=np.linspace(0, 1, 21), alpha=.55, label=label, color=color)
    axes[0].axvline(.5, color='gray', lw=1)
    axes[0].set(xlabel='CNN estimated c on held-out observed shapes', ylabel='Scans'); axes[0].legend()
    for pair in range(pseudo['pairs']):
        selected = [r for r in pseudo['records'] if r['pair_index'] == pair]
        axes[1].plot([r['c'] for r in selected], [pseudo_predictions[r['index']] for r in selected], marker='.',
                     label=f"{selected[0]['pid']} CN:{selected[0]['source_uid']}")
    axes[1].plot([0, 1], [0, 1], lw=1, color='gray', label='Synthetic target c')
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Train pseudo label c', ylabel='CNN estimated c'); axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(report/'cnn/cognition.png', dpi=180); fig.savefig(report/'cnn/cognition.pdf'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
