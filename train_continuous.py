"""Frozen train-only Eq.4 sampling, cognition learning and Algorithm 2 evaluation.

The paper does not specify sampling density, CNN optimization, augmentation
mixing, fine-tuning duration or progressive step count. The experiment config
records those supplements. Observed-data and third-party snapshots are immutable.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import open3d as o3d
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import trimesh

from cognition import CognitionEstimator3D, PCAShapeVoxelizer, RK4ShapeIntegrator
from evaluation import evaluate_pca_prediction
from deterministic_pool import DisjointMaxPool3D
from lab_shapes import ShapePCA, patient_indices
from model import ODEFuncWithAttention
from paper_cognition import pseudo_cognitive_sample
from paper_training import optimizer_batch_step, scale_pca_shapes, trajectory_loss
from train_lab import aggregate, atomic_json, digest, predict_future

PROJECT = Path(__file__).resolve().parent


def conversion_pairs(data):
    """All strictly later CN->AD endpoint combinations, training patients only."""
    pairs = []
    for pid, ids in patient_indices(data['pids'], data['splits']).items():
        if data['splits'][ids[0]] != 'train':
            continue
        for source in ids:
            for target in ids:
                if (data['conditions'][source] == 0 and data['conditions'][target] == 1
                        and data['times'][target] > data['times'][source]):
                    pairs.append((pid, int(source), int(target)))
    return pairs


def mesh_qc(vertices, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(faces))
    intersections = np.asarray(mesh.get_self_intersecting_triangles())
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1)/2
    affected = np.unique(intersections) if intersections.size else np.array([], dtype=int)
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    volume = float(tri.volume)
    if not tri.is_watertight or not np.isfinite(volume) or volume <= 0:
        raise ValueError('nonwatertight or nonpositive signed-volume mesh')
    return dict(self_intersection_pairs=len(intersections),
                affected_area_percent=float(100*areas[affected].sum()/areas.sum()),
                signed_mesh_volume_mm3=volume)


class Experiment:
    def __init__(self, path, stage):
        self.path = path.resolve()
        self.config = json.loads(self.path.read_text())
        self.stage = stage
        self.report = Path(self.config['report'])/stage
        self.output = Path(self.config['output'])/stage
        self.report.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.status = dict(stage=stage, status='preflight', pid=os.getpid(),
                           config_sha256=digest(self.path))
        self.update()
        for name, expected in self.config['code_sha256'].items():
            if digest(PROJECT/name) != expected:
                raise ValueError('frozen code changed: '+name)
        for name, expected in self.config['data_sha256'].items():
            if digest(Path(self.config['data_root'])/name) != expected:
                raise ValueError('frozen source data changed: '+name)
        if digest(self.config['initial_checkpoint']) != self.config['initial_checkpoint_sha256']:
            raise ValueError('initial observed checkpoint changed')
        if digest(self.config['voxel_manifest']) != self.config['voxel_manifest_sha256']:
            raise ValueError('observed voxel manifest changed')
        with np.load(Path(self.config['data_root'])/'observations.npz', allow_pickle=False) as archive:
            self.data = {k: archive[k].copy() for k in archive.files}
        self.pca = ShapePCA.load(Path(self.config['data_root'])/'pca.npz')
        self.groups = patient_indices(self.data['pids'], self.data['splits'])
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.set_num_threads(1 if stage == 'ode' else 4)
        torch.cuda.manual_seed_all(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            raise ValueError('configured GPU unavailable')
        grid = self.config['grid']
        self.voxel3 = PCAShapeVoxelizer(self.pca, self.data['faces'], **grid, occupancy_rays=3)
        self.voxel7 = PCAShapeVoxelizer(self.pca, self.data['faces'], **grid, occupancy_rays=7)

    def update(self, **fields):
        self.status.update(fields, elapsed_seconds=time.monotonic()-self.started,
                           updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.report/'progress.json', self.status)

    def fresh_output(self):
        self.output.mkdir(parents=True, exist_ok=False)

    def ode_model(self, checkpoint):
        model = ODEFuncWithAttention(150, 1).to(self.device)
        # These are locally generated, checksum-verified experiment checkpoints.
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        model.load_state_dict(saved['model'])
        model.eval()
        return model

    def pseudo(self):
        root = Path(self.config['output'])/'pseudo'
        manifest = json.loads((root/'manifest.json').read_text())
        if digest(root/'samples.npz') != manifest['samples_sha256']:
            raise ValueError('pseudo samples changed')
        with np.load(root/'samples.npz', allow_pickle=False) as file:
            arrays = {k: file[k].copy() for k in file.files}
        if any(self.data['splits'][r['source_index']] != 'train' or
               self.data['splits'][r['target_index']] != 'train' for r in manifest['records']):
            raise ValueError('test endpoint in pseudo training data')
        return arrays, manifest

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        tmp = self.output/'latest.pt.tmp'
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), config=self.config,
                        config_sha256=digest(self.path), random_state=random.getstate(),
                        numpy_state=np.random.get_state(), torch_rng_state=torch.get_rng_state(),
                        cuda_rng_state=torch.cuda.get_rng_state_all()), tmp)
        tmp.replace(self.output/'latest.pt')

    def environment(self, model):
        atomic_json(self.report/'environment.json', dict(torch=torch.__version__, numpy=np.__version__,
            gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            parameter_count=sum(p.numel() for p in model.parameters()), fp32=True, tf32=False))


def prepare(experiment):
    ex = experiment
    ex.fresh_output()
    pairs = conversion_pairs(ex.data)
    if len(pairs) != ex.config['conversion_pairs']:
        raise ValueError('conversion endpoint count changed')
    model = ex.ode_model(ex.config['initial_checkpoint'])
    integrate = RK4ShapeIntegrator(model)
    coefficients, records = [], []
    with torch.inference_mode():
        for pair_index, (pid, source, target) in enumerate(pairs):
            for fraction in ex.config['pseudo_fractions']:
                sample_time = float(ex.data['times'][source]+fraction*(ex.data['times'][target]-ex.data['times'][source]))
                value, c = pseudo_cognitive_sample(ex.data['coefficients'][source], ex.data['coefficients'][target],
                    ex.data['times'][source], ex.data['times'][target], sample_time, integrate=integrate)
                coefficients.append(value.cpu().numpy())
                records.append(dict(index=len(records), pair_index=pair_index, pid=pid, split='train',
                    source_index=source, target_index=target, source_uid=str(ex.data['uids'][source]),
                    target_uid=str(ex.data['uids'][target]), time=sample_time, c=c,
                    file=str(ex.output/f'p{pair_index:02d}_c{fraction:.1f}.npz')))
    coefficients = np.stack(coefficients)
    vertices = ex.pca.inverse_transform(coefficients).reshape(-1, len(ex.data['vertices'][0]), 3)
    ex.update(status='voxel_and_surface_qc', samples=len(records))

    def check(record):
        i = record['index']
        result = dict(record, **mesh_qc(vertices[i], ex.data['faces']))
        first, second = ex.voxel3(coefficients[i]), ex.voxel7(coefficients[i])
        disagreement = int(np.count_nonzero(first != second))
        if disagreement:
            raise ValueError('3/7-ray pseudo voxelization disagreement')
        np.savez_compressed(result['file'], occupancy=first[0, 0].astype(np.uint8))
        result.update(sha256=digest(result['file']), occupied_voxels=int(first.sum()),
                      ray_vote_disagreement_voxels=disagreement,
                      voxel_to_signed_volume_bias_percent=float(100*(first.sum()*.125/result['signed_mesh_volume_mm3']-1)))
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(check, records))
    np.savez_compressed(ex.output/'samples.npz', coefficients=coefficients, vertices=vertices,
        faces=ex.data['faces'], times=np.array([r['time'] for r in records]),
        conditions=np.array([r['c'] for r in records]), pids=np.array([r['pid'] for r in records]),
        pair_indices=np.array([r['pair_index'] for r in records]))
    manifest = dict(records=records, samples_sha256=digest(ex.output/'samples.npz'),
        initial_checkpoint_sha256=ex.config['initial_checkpoint_sha256'], formula='paper Eq.4; half forward CN + half backward AD',
        subjects=len({r['pid'] for r in records}), pairs=len(pairs), samples=len(records),
        self_intersecting_samples=sum(r['self_intersection_pairs'] > 0 for r in records),
        max_affected_area_percent=max(r['affected_area_percent'] for r in records),
        max_abs_voxel_signed_volume_bias_percent=max(abs(r['voxel_to_signed_volume_bias_percent']) for r in records),
        excluded_or_repaired=0, test_endpoints_used=0, limitation='Signed-volume agreement does not validate anatomical interiors.')
    atomic_json(ex.output/'manifest.json', manifest)
    atomic_json(ex.report/'manifest.json', manifest)
    ex.update(status='completed', **{k: manifest[k] for k in ('subjects', 'pairs', 'samples',
        'self_intersecting_samples', 'max_affected_area_percent', 'max_abs_voxel_signed_volume_bias_percent')})


def load_voxels(records):
    voxels = np.empty((len(records), 1, 128, 128, 128), dtype=np.uint8)
    for i, record in enumerate(records):
        if digest(record['file']) != record['sha256']:
            raise ValueError('voxel file changed: '+record['file'])
        with np.load(record['file'], allow_pickle=False) as archive:
            voxels[i, 0] = archive['occupancy']
    return torch.from_numpy(voxels)


def cognition_metrics(rows):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
    y = np.array([r['observed_c'] for r in rows], dtype=int)
    p = np.array([r['estimated_c'] for r in rows])
    return dict(scans=len(rows), patients=len({r['pid'] for r in rows}),
        accuracy=float(accuracy_score(y, p >= .5)),
        balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
        roc_auc=float(roc_auc_score(y, p)), confusion_matrix=confusion_matrix(y, p >= .5, labels=[0, 1]).tolist(),
        brier=float(np.mean((p-y)**2)), mean_c_CN=float(p[y == 0].mean()) if np.any(y == 0) else None,
        mean_c_AD=float(p[y == 1].mean()) if np.any(y == 1) else None)


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cognition_model(ex):
    model = CognitionEstimator3D().to(ex.device)
    if ex.config['cnn'].get('pooling_backend') == 'deterministic-disjoint-max':
        for index, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.MaxPool3d):
                model.features[index] = DisjointMaxPool3D()
    return model


def train_cnn(ex):
    ex.fresh_output()
    pseudo, pseudo_manifest = ex.pseudo()
    observed = json.loads(Path(ex.config['voxel_manifest']).read_text())['records']
    training = [r for r in observed if r['split'] == 'train'] + pseudo_manifest['records']
    labels = torch.tensor([r.get('observed_c', r.get('c')) for r in training], dtype=torch.float32)
    voxels = load_voxels(training)
    settings = ex.config['cnn']
    loader = DataLoader(TensorDataset(voxels, labels), batch_size=settings['batch_size'], shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=False,
                        generator=torch.Generator().manual_seed(ex.config['seed']))
    model = cognition_model(ex)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)
    ex.environment(model)
    ex.update(status='training', observed_training_scans=1150, pseudo_training_samples=len(pseudo['coefficients']),
              samples_per_epoch=len(training), epochs_completed=0, planned_epochs=settings['epochs'])
    log = ex.report/'epochs.csv'
    with log.open('w', newline='', buffering=1) as file:
        fields = ['epoch', 'train_soft_bce', 'observed_bce', 'pseudo_soft_bce', 'learning_rate', 'epoch_seconds']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, settings['epochs']+1):
            started = time.monotonic()
            model.train()
            total, real_total, pseudo_total, real_count, pseudo_count = 0., 0., 0., 0, 0
            lr = optimizer.param_groups[0]['lr']
            for x, y in loader:
                x = x.to(ex.device, dtype=torch.float32, non_blocking=True)
                y = y.to(ex.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                losses = F.binary_cross_entropy_with_logits(model.logits(x), y, reduction='none')
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite CNN training loss')
                loss.backward()
                optimizer.step()
                is_real = (y == 0) | (y == 1)
                total += float(losses.sum().detach())
                real_total += float(losses[is_real].sum().detach())
                pseudo_total += float(losses[~is_real].sum().detach())
                real_count += int(is_real.sum()); pseudo_count += int((~is_real).sum())
            scheduler.step()
            row = dict(epoch=epoch, train_soft_bce=total/len(training), observed_bce=real_total/real_count,
                       pseudo_soft_bce=pseudo_total/pseudo_count, learning_rate=lr, epoch_seconds=time.monotonic()-started)
            writer.writerow(row); file.flush(); print(json.dumps(row), flush=True)
            ex.save_checkpoint(model, optimizer, scheduler, epoch)
            ex.update(epochs_completed=epoch, last_epoch=row,
                estimated_remaining_seconds=(time.monotonic()-ex.started)/epoch*(settings['epochs']-epoch))
    del loader, voxels
    ex.update(status='evaluating_fixed_final_checkpoint')
    model.eval()
    # Evaluation happens once, after fixed training. No test-based stopping or selection.
    evaluation_records = observed + pseudo_manifest['records']
    voxels = load_voxels(evaluation_records)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(voxels), settings['batch_size']):
            predictions.extend(model(voxels[start:start+settings['batch_size']].to(ex.device, dtype=torch.float32)).cpu().tolist())
    rows = [dict(pid=r['pid'], uid=r.get('uid', f"pseudo:{r.get('index')}"), split=r['split'],
        observed_c=float(r.get('observed_c', r.get('c'))), estimated_c=float(p),
        original_quality_flags=r.get('original_quality_flags', ''), pseudo='uid' not in r)
        for r, p in zip(evaluation_records, predictions)]
    write_csv(ex.report/'cognition_predictions.csv', rows)
    test = [r for r in rows if r['split'] == 'test' and not r['pseudo']]
    train = [r for r in rows if r['split'] == 'train' and not r['pseudo']]
    pseudo_rows = [r for r in rows if r['pseudo']]
    result = dict(test=cognition_metrics(test), train=cognition_metrics(train),
        pseudo_training_mae=float(np.mean([abs(r['observed_c']-r['estimated_c']) for r in pseudo_rows])),
        selection='fixed final epoch; no test selection', future_or_test_pseudo_supervision=False,
        continuous_ground_truth_available=False,
        limitation='Observed diagnosis probability and synthetic fractional labels; not measured clinical cognition severity.',
        checkpoint_sha256=digest(ex.output/'latest.pt'))
    atomic_json(ex.report/'evaluation.json', result)
    ex.update(status='completed', evaluation=result, checkpoint_sha256=result['checkpoint_sha256'], estimated_remaining_seconds=0)


def augmented_trajectories(data, pseudo, manifest):
    """Keep different endpoint-relative c trajectories separate within a patient."""
    by_patient = {}
    for pair in sorted(set(pseudo['pair_indices'].tolist())):
        ids = np.flatnonzero(pseudo['pair_indices'] == pair)
        record = manifest['records'][int(ids[0])]
        source, target = record['source_index'], record['target_index']
        if data['pids'][source] != data['pids'][target] or data['splits'][source] != 'train' or data['splits'][target] != 'train':
            raise ValueError('invalid pseudo endpoints')
        shapes = np.concatenate([data['coefficients'][source:source+1], pseudo['coefficients'][ids],
                                 data['coefficients'][target:target+1]])
        times = np.concatenate([data['times'][source:source+1], pseudo['times'][ids], data['times'][target:target+1]])
        conditions = np.concatenate([[0.], pseudo['conditions'][ids], [1.]])
        if not np.all(np.diff(times) > 0):
            raise ValueError('pseudo trajectory must be strictly ordered')
        by_patient.setdefault(record['pid'], []).append((shapes, times, conditions))
    return by_patient


def train_ode(ex):
    ex.fresh_output()
    pseudo, manifest = ex.pseudo()
    augmented = augmented_trajectories(ex.data, pseudo, manifest)
    train = {pid: ids for pid, ids in ex.groups.items() if ex.data['splits'][ids[0]] == 'train'
             and len(np.unique(ex.data['times'][ids])) > 1}
    model = ex.ode_model(ex.config['initial_checkpoint'])
    settings = ex.config['ode']
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.5)
    mean = torch.tensor(ex.pca.mean_, dtype=torch.float32, device=ex.device)
    components = torch.tensor(ex.pca.components_, dtype=torch.float32, device=ex.device)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=ex.device)
    tensors = {pid: tuple(tensor(ex.data[name][ids]) for name in ('coefficients', 'times', 'conditions')) for pid, ids in train.items()}
    augment_tensors = {pid: [tuple(tensor(value) for value in trajectory) for trajectory in trajectories]
                      for pid, trajectories in augmented.items()}
    ex.environment(model)
    ex.update(status='training', epochs_completed=0, planned_epochs=settings['epochs'],
              temporal_training_patients=len(train), augmented_patients=len(augmented), pseudo_trajectories=len(conversion_pairs(ex.data)))
    with (ex.report/'epochs.csv').open('w', newline='', buffering=1) as file:
        fields = ['epoch', 'train_patient_mean_squared_pca_l2', 'real_patient_mean_loss',
                  'pseudo_trajectory_mean_loss', 'ordered_pairs', 'optimizer_steps', 'learning_rate', 'epoch_seconds']
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for epoch in range(1, settings['epochs']+1):
            started = time.monotonic()
            model.train(); order = list(train); random.shuffle(order)
            pending, values, real_values, pseudo_values = [], [], [], []
            pairs, steps = 0, 0
            lr = optimizer.param_groups[0]['lr']
            for pid in order:
                coefficients, times, conditions = tensors[pid]
                factor = random.uniform(.98, 1.02)
                loss, count = trajectory_loss(model, scale_pca_shapes(coefficients, mean, components, factor), times, conditions)
                real_values.append(float(loss.detach())); pairs += count
                if pid in augment_tensors:
                    intermediate_losses = []
                    for shapes, pseudo_times, pseudo_conditions in augment_tensors[pid]:
                        part, count = trajectory_loss(model, scale_pca_shapes(shapes, mean, components, factor), pseudo_times, pseudo_conditions)
                        intermediate_losses.append(part); pseudo_values.append(float(part.detach())); pairs += count
                    loss = .5*loss + .5*torch.stack(intermediate_losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite augmented ODE loss')
                values.append(float(loss.detach())); pending.append(loss)
                if len(pending) == settings['batch_subjects']:
                    optimizer_batch_step(optimizer, pending); pending = []; steps += 1
            if pending:
                optimizer_batch_step(optimizer, pending); steps += 1
            scheduler.step()
            row = dict(epoch=epoch, train_patient_mean_squared_pca_l2=float(np.mean(values)),
                       real_patient_mean_loss=float(np.mean(real_values)), pseudo_trajectory_mean_loss=float(np.mean(pseudo_values)),
                       ordered_pairs=pairs, optimizer_steps=steps, learning_rate=lr, epoch_seconds=time.monotonic()-started)
            writer.writerow(row); file.flush(); print(json.dumps(row), flush=True)
            ex.save_checkpoint(model, optimizer, scheduler, epoch)
            ex.update(epochs_completed=epoch, last_epoch=row,
                      estimated_remaining_seconds=(time.monotonic()-ex.started)/epoch*(settings['epochs']-epoch))
    ex.update(status='completed', checkpoint_sha256=digest(ex.output/'latest.pt'), estimated_remaining_seconds=0)


def evaluate(ex):
    """Algorithm 2 across independent target trajectories; no cross-patient attention."""
    ex.fresh_output()
    for stage in ('cnn', 'ode'):
        progress = json.loads((Path(ex.config['report'])/stage/'progress.json').read_text())
        if progress['status'] != 'completed' or digest(Path(ex.config['output'])/stage/'latest.pt') != progress['checkpoint_sha256']:
            raise ValueError('training stage incomplete or checkpoint changed: '+stage)
    model = ex.ode_model(Path(ex.config['output'])/'ode/latest.pt')
    cognition = cognition_model(ex)
    saved = torch.load(Path(ex.config['output'])/'cnn/latest.pt', map_location=ex.device, weights_only=False)
    cognition.load_state_dict(saved['model']); cognition.eval()
    integrate = RK4ShapeIntegrator(model)
    entries = []
    for pid, ids in ex.groups.items():
        if ex.data['splits'][ids[0]] != 'test':
            continue
        for target in ids:
            if ex.data['times'][target] > ex.data['times'][ids[0]]:
                entries.append(dict(pid=pid, source=ids[0], target=target))
    if len(entries) != ex.config['test_future_targets']:
        raise ValueError('future target count mismatch')
    states = [torch.tensor(ex.data['coefficients'][r['source']], dtype=torch.float32, device=ex.device) for r in entries]
    steps = ex.config['progressive_steps']
    step_rows = []
    ex.update(status='progressive_forecasting', targets=len(entries), planned_steps=steps, steps_completed=0)

    def voxel_check(item):
        index, value = item
        vertices = ex.pca.inverse_transform(value[None]).reshape(-1, 3)
        geometry = mesh_qc(vertices, ex.data['faces'])
        first, second = ex.voxel3(value), ex.voxel7(value)
        disagreement = int(np.count_nonzero(first != second))
        if disagreement:
            raise ValueError('3/7-ray intermediate voxel disagreement')
        geometry.update(ray_vote_disagreement_voxels=disagreement,
            voxel_to_signed_volume_bias_percent=float(100*(first.sum()*.125/geometry['signed_mesh_volume_mm3']-1)),
            occupied_voxels=int(first.sum()))
        return index, first[0, 0].astype(np.uint8), geometry

    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for step in range(steps):
            values = [value.cpu().numpy() for value in states]
            voxels, checks = [], []
            for index, voxel, geometry in pool.map(voxel_check, enumerate(values)):
                voxels.append(voxel); checks.append(geometry)
            conditions = []
            batch = ex.config['cnn']['batch_size']
            for start in range(0, len(voxels), batch):
                x = torch.from_numpy(np.stack(voxels[start:start+batch])[:, None]).to(ex.device, dtype=torch.float32)
                conditions.extend(cognition(x).cpu().tolist())
            del voxels
            for index, (entry, c, geometry) in enumerate(zip(entries, conditions, checks)):
                source, target = entry['source'], entry['target']
                initial, final = float(ex.data['times'][source]), float(ex.data['times'][target])
                a = initial + step*(final-initial)/steps
                b = final if step == steps-1 else initial+(step+1)*(final-initial)/steps
                states[index] = integrate(states[index], a, b, c)
                step_rows.append(dict(pid=entry['pid'], source_uid=str(ex.data['uids'][source]),
                    target_uid=str(ex.data['uids'][target]), step=step+1, source_time=a, target_time=b, estimated_c=float(c), **geometry))
            write_csv(ex.report/'intermediate_states.csv', step_rows)
            ex.update(steps_completed=step+1, intermediate_inputs_checked=len(step_rows))
            print(json.dumps(ex.status), flush=True)
        predicted = [state.cpu().numpy() for state in states]
        # Fixed observed-c fine-tuned control isolates progressive-estimator behavior.
        constant = {}
        for pid, ids in ex.groups.items():
            if ex.data['splits'][ids[0]] != 'test':
                continue
            target_indices, predictions = predict_future(model,
                torch.tensor(ex.data['coefficients'][ids], dtype=torch.float32, device=ex.device),
                torch.tensor(ex.data['times'][ids], dtype=torch.float32, device=ex.device),
                torch.tensor(ex.data['conditions'][ids[0]], dtype=torch.float32, device=ex.device))
            for local, value in zip(target_indices.cpu().tolist(), predictions.cpu().numpy()):
                constant[int(ids[local])] = value
    rows, meshes = [], []
    for entry, value in zip(entries, predicted):
        source, target = entry['source'], entry['target']
        metrics = evaluate_pca_prediction(ex.pca, value, ex.data['vertices'][target], ex.data['vertices'][source])
        vertices = metrics.pop('predicted_vertices'); metrics.pop('per_vertex_euclidean')
        meshes.append(vertices)
        control = evaluate_pca_prediction(ex.pca, constant[target], ex.data['vertices'][target], ex.data['vertices'][source])
        geometry = mesh_qc(vertices, ex.data['faces'])
        # The final output also passes fixed-grid checks, not only Algorithm 2 inputs.
        _, _, final_voxel = voxel_check((0, value))
        row = dict(pid=entry['pid'], source_uid=str(ex.data['uids'][source]), target_uid=str(ex.data['uids'][target]),
            source_day=str(ex.data['days'][source]), target_day=str(ex.data['days'][target]),
            horizon_years=float(ex.data['ages'][target]-ex.data['ages'][source]),
            target_time=float(ex.data['times'][target]), source_flags=str(ex.data['qc_flags'][source]),
            target_flags=str(ex.data['qc_flags'][target]), **metrics,
            augmented_fixed_c_euclidean=control['prediction_mean_vertex_euclidean'],
            prediction_self_intersection_pairs=geometry['self_intersection_pairs'],
            prediction_affected_area_percent=geometry['affected_area_percent'],
            final_voxel_signed_volume_bias_percent=final_voxel['voxel_to_signed_volume_bias_percent'])
        rows.append(row)
    write_csv(ex.report/'future_metrics.csv', rows)
    write_csv(ex.report/'intermediate_states.csv', step_rows)
    np.savez_compressed(ex.output/'future_predictions.npz', coefficients=np.stack(predicted), vertices=np.stack(meshes),
        fixed_c_coefficients=np.stack([constant[r['target']] for r in entries]), faces=ex.data['faces'],
        pids=np.array([r['pid'] for r in entries]), source_uids=np.array([r['source_uid'] for r in rows]),
        target_uids=np.array([r['target_uid'] for r in rows]))
    latest = [r for r in rows if r['target_time'] == max(other['target_time'] for other in rows if other['pid'] == r['pid'])]
    write_csv(ex.report/'latest_visit_metrics.csv', latest)
    result = dict(all=aggregate(rows), latest=aggregate(latest),
        checkpoint_selection='fixed final epochs; no held-out selection',
        progressive_steps=steps, future_diagnosis_supplied=False,
        predictions_with_self_intersection=sum(r['prediction_self_intersection_pairs'] > 0 for r in rows),
        max_predicted_affected_area_percent=max(r['prediction_affected_area_percent'] for r in rows),
        intermediate_inputs_with_self_intersection=sum(r['self_intersection_pairs'] > 0 for r in step_rows),
        max_intermediate_affected_area_percent=max(r['affected_area_percent'] for r in step_rows),
        max_intermediate_voxel_signed_volume_bias_percent=max(abs(r['voxel_to_signed_volume_bias_percent']) for r in step_rows),
        ray_vote_disagreements=sum(r['ray_vote_disagreement_voxels'] for r in step_rows),
        estimated_c_range=[min(r['estimated_c'] for r in step_rows), max(r['estimated_c'] for r in step_rows)],
        prediction_file_sha256=digest(ex.output/'future_predictions.npz'),
        units='assumed mm; source spatial units unspecified',
        limitation='Two training conversion patients and no held-out CN->AD converters; geometry consistency is not anatomical validation.')
    atomic_json(ex.report/'evaluation.json', result)
    ex.update(status='completed', evaluation=result, estimated_remaining_seconds=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['pseudo', 'cnn', 'ode', 'evaluate'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    ex = Experiment(args.config, args.stage)
    try:
        dict(pseudo=prepare, cnn=train_cnn, ode=train_ode, evaluate=evaluate)[args.stage](ex)
    except BaseException as error:
        ex.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
