"""Predeclared N=10 fixed-condition controls for Algorithm 2 evaluation.

Controls share the progressive RK4 time grid, first-observation source, PCA,
targets and final checkpoints. They separate cognition updates from numerical
solver segmentation. Augmentation versus initial training still combines extra
optimization and pseudo supervision; it is not a pseudo-only ablation.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cognition import RK4ShapeIntegrator
from evaluation import evaluate_pca_prediction
from lab_shapes import ShapePCA
from model import ODEFuncWithAttention
from train_lab import aggregate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def main():
    report = PROJECT/'artifacts/training_runs/adni1_continuous_v1'
    config = json.loads((report/'deterministic_pool_config.json').read_text())
    output = Path(config['output'])
    progress = json.loads((report/'evaluate/progress.json').read_text())
    if progress['status'] != 'completed':
        raise ValueError('completed Algorithm 2 evaluation required')
    steps = config['progressive_steps']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    checkpoints = [Path(config['initial_checkpoint']), output/'ode/latest.pt']
    expected = [config['initial_checkpoint_sha256'], json.loads((report/'ode/progress.json').read_text())['checkpoint_sha256']]
    solvers = []
    for path, value in zip(checkpoints, expected):
        if digest(path) != value:
            raise ValueError('checkpoint changed')
        model = ODEFuncWithAttention(150, 1).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
        model.eval(); solvers.append(RK4ShapeIntegrator(model))
    with np.load(Path(config['data_root'])/'observations.npz', allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    pca = ShapePCA.load(Path(config['data_root'])/'pca.npz')
    indices = {str(uid): i for i, uid in enumerate(data['uids'])}
    metrics = read_rows(report/'evaluate/future_metrics.csv')
    first_c = {r['target_uid']: float(r['estimated_c']) for r in read_rows(report/'evaluate/intermediate_states.csv') if int(r['step']) == 1}
    controls = ['initial_observed_c_matched_euclidean', 'augmented_observed_c_matched_euclidean', 'augmented_cnn_initial_c_matched_euclidean']
    rows, values = [], []
    with torch.inference_mode():
        for metric in metrics:
            source, target = indices[metric['source_uid']], indices[metric['target_uid']]
            initial, final = float(data['times'][source]), float(data['times'][target])
            sample = dict(pid=metric['pid'], source_uid=metric['source_uid'], target_uid=metric['target_uid'],
                          target_time=final, algorithm2_euclidean=float(metric['prediction_mean_vertex_euclidean']))
            coefficient_controls = []
            for key, solver, c in zip(controls, [solvers[0], solvers[1], solvers[1]],
                                      [data['conditions'][source], data['conditions'][source], first_c[metric['target_uid']]]):
                state = torch.tensor(data['coefficients'][source], dtype=torch.float32, device=device)
                for step in range(steps):
                    a = initial+step*(final-initial)/steps
                    b = final if step == steps-1 else initial+(step+1)*(final-initial)/steps
                    state = solver(state, a, b, c)
                value = state.cpu().numpy(); coefficient_controls.append(value)
                sample[key] = evaluate_pca_prediction(pca, value, data['vertices'][target], data['vertices'][source])['prediction_mean_vertex_euclidean']
            rows.append(sample); values.append(coefficient_controls)
    with (report/'evaluate/matched_controls.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output/'evaluate/matched_controls.npz', coefficients=np.asarray(values),
        names=np.asarray(controls), source_uids=np.asarray([r['source_uid'] for r in rows]),
        target_uids=np.asarray([r['target_uid'] for r in rows]))
    latest = [r for r in rows if r['target_time'] == max(s['target_time'] for s in rows if s['pid'] == r['pid'])]
    result = dict(all=aggregate(rows), latest=aggregate(latest), controls=controls, steps=steps,
        future_diagnosis_supplied=False, selection='predeclared controls; fixed final checkpoints and step count',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), checkpoints_sha256=expected,
        code_sha256=digest(Path(__file__)), predictions_sha256=digest(output/'evaluate/matched_controls.npz'),
        limitation='Augmented-versus-initial comparison combines extra optimization and pseudo supervision.')
    (report/'evaluate/matched_controls.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
