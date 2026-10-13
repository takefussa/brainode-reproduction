"""Check age/volume distributions and exact voxel duplication across splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    project = Path(__file__).resolve().parents[1]
    report = project/'artifacts/training_runs/adni1_continuous_v1'
    config = json.loads((report/'config.json').read_text())
    observed = json.loads(Path(config['voxel_manifest']).read_text())['records']
    patient_sets = {split: {r['pid'] for r in observed if r['split'] == split} for split in ('train', 'test')}
    patient_overlap = len(patient_sets['train'] & patient_sets['test'])
    if patient_overlap:
        raise ValueError('patient occurs in both train and test')
    with np.load(Path(config['data_root'])/'observations.npz', allow_pickle=False) as file:
        ages = file['ages'].copy()
    fingerprints = {}
    for record in observed:
        with np.load(record['file'], allow_pickle=False) as file:
            key = hashlib.sha256(file['occupancy'].tobytes()).hexdigest()
        fingerprints.setdefault(key, []).append(record)
    cross_split = [group for group in fingerprints.values() if len({r['split'] for r in group}) > 1]
    distributions = {}
    for split in ('train', 'test'):
        records = [r for r in observed if r['split'] == split]
        y = np.array([r['observed_c'] for r in records])
        volume = np.array([r['pca_mesh_volume_mm3'] for r in records])
        groups = {}
        for label, name in ((0, 'CN'), (1, 'AD')):
            selected = [r for r in records if r['observed_c'] == label]
            groups[name] = dict(scans=len(selected), patients=len({r['pid'] for r in selected}),
                age_mean=float(np.mean([ages[r['index']] for r in selected])),
                age_range=[float(min(ages[r['index']] for r in selected)), float(max(ages[r['index']] for r in selected))],
                pca_signed_volume_mean=float(np.mean([r['pca_mesh_volume_mm3'] for r in selected])))
        distributions[split] = dict(by_diagnosis=groups, smaller_volume_AD_ranking_auc=float(roc_auc_score(y, -volume)))
    result = dict(patient_split_overlap=patient_overlap, exact_occupancy_groups_across_train_test=len(cross_split),
        duplicate_groups=[dict(pids=[r['pid'] for r in g], uids=[r['uid'] for r in g]) for g in cross_split],
        distributions=distributions, selection_or_model_changed=False,
        limitation='Volume ranking is a descriptive baseline without fitted classifier or test-based parameter choice. Distribution differences do not fully explain performance differences.',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if cross_split:
        raise ValueError('exact occupied grid duplicated across training and test patients')
    (report/'cnn/distribution_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
