"""Observed-data BrainODE baseline with frozen laboratory data and settings."""

import argparse
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
from torchdiffeq import odeint

from evaluation import evaluate_pca_prediction
from lab_shapes import ShapePCA, patient_indices
from model import ODEFuncWithAttention
from paper_training import optimizer_batch_step, scale_pca_shapes, trajectory_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def predict_future(model, coefficients, times, initial_c):
    """Use the first observation and its c only; repeat targets remain separate."""
    targets=torch.nonzero(times>times[0]).flatten()
    if not len(targets):
        return targets, coefficients.new_empty((0,coefficients.shape[1]))
    unique,mapping=torch.unique(times[targets],sorted=True,return_inverse=True)
    solver_times=torch.cat([times[:1],unique])
    prediction=odeint(lambda t,value:model(t,value,initial_c.reshape(1)),
                       coefficients[:1],solver_times,method='rk4')[1:,0]
    return targets,prediction[mapping]


def aggregate(rows):
    if not rows:
        return dict(subjects=0,targets=0)
    keys=[k for k in rows[0] if k.endswith('_euclidean') or k=='public_coordinate_mae_to_pca_target']
    groups={}
    for row in rows:
        groups.setdefault(row['pid'],[]).append(row)
    return dict(subjects=len(groups),targets=len(rows),
        subject_macro_mean={k:float(np.mean([np.mean([r[k] for r in group]) for group in groups.values()])) for k in keys},
        target_micro_mean={k:float(np.mean([r[k] for r in rows])) for k in keys})


def evaluate(model, data, pca, groups, device, output, report):
    model.eval()
    rows, predicted_meshes=[],[]
    with torch.no_grad():
        for pid,indices in groups.items():
            if data['splits'][indices[0]]!='test':
                continue
            coefficients=torch.as_tensor(data['coefficients'][indices],dtype=torch.float32,device=device)
            times=torch.as_tensor(data['times'][indices],dtype=torch.float32,device=device)
            initial_c=torch.tensor(data['conditions'][indices[0]],dtype=torch.float32,device=device)
            targets,predicted=predict_future(model,coefficients,times,initial_c)
            for local_index,prediction in zip(targets.cpu().tolist(),predicted.cpu().numpy()):
                source,target=indices[0],indices[local_index]
                metrics=evaluate_pca_prediction(pca,prediction,data['vertices'][target],data['vertices'][source])
                mesh_vertices=metrics.pop('predicted_vertices')
                predicted_meshes.append(mesh_vertices)
                metrics.pop('per_vertex_euclidean')
                mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_vertices),
                                               o3d.utility.Vector3iVector(data['faces']))
                metrics['prediction_self_intersection_pairs']=len(mesh.get_self_intersecting_triangles())
                rows.append(dict(pid=pid,source_uid=str(data['uids'][source]),target_uid=str(data['uids'][target]),
                    source_day=str(data['days'][source]),target_day=str(data['days'][target]),
                    horizon_years=float(data['ages'][target]-data['ages'][source]),
                    initial_observed_c=float(data['conditions'][source]),
                    source_flags=str(data['qc_flags'][source]),target_flags=str(data['qc_flags'][target]),**metrics))
    if not rows:
        raise ValueError('no held-out future targets')
    with (report/'future_metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'future_predictions.npz',vertices=np.stack(predicted_meshes),faces=data['faces'],
        pids=np.array([r['pid'] for r in rows]),source_uids=np.array([r['source_uid'] for r in rows]),
        target_uids=np.array([r['target_uid'] for r in rows]))
    summaries=dict(all=aggregate(rows),
        target_flagged=aggregate([r for r in rows if r['target_flags']]),
        target_unflagged=aggregate([r for r in rows if not r['target_flags']]),
        source_or_target_flagged=aggregate([r for r in rows if r['source_flags'] or r['target_flags']]),
        source_and_target_unflagged=aggregate([r for r in rows if not(r['source_flags'] or r['target_flags'])]))
    result=dict(evaluation='first chronological observation to all strictly later-time test observations',
        checkpoint_selection='fixed final epoch; test not used for training or checkpoint selection',
        primary_aggregation='equal-weight mean over patients after averaging their future targets',
        metrics=summaries,future_diagnosis_supplied=False,cnn_or_algorithm2_used=False,
        predicted_targets_with_self_intersection=sum(r['prediction_self_intersection_pairs']>0 for r in rows),
        units='assumed mm; numerical source units unspecified',
        prediction_file_sha256=digest(output/'future_predictions.npz'))
    atomic_json(report/'evaluation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config_path=args.config.resolve()
    config=json.loads(config_path.read_text())
    report,output=Path(config['report']),Path(config['output'])
    if not output.is_relative_to(Path('/data5/take/brainode')) or not report.is_relative_to(Path('/home/take/brainode')):
        raise ValueError('experiment paths outside the configured storage roots')
    report.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status=dict(status='preflight',pid=os.getpid(),epochs_completed=0,planned_epochs=config['epochs'],
        config_sha256=digest(config_path),output=str(output),report=str(report))
    def update(**fields):
        status.update(fields,elapsed_seconds=time.monotonic()-started,
            updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(report/'progress.json',status)
    update()
    try:
        for name,expected in config['code_sha256'].items():
            if digest(Path(__file__).resolve().parent/name)!=expected:
                raise ValueError('frozen experiment code changed: '+name)
        data_root=Path(config['data_root'])
        for name,expected in config['data_sha256'].items():
            if digest(data_root/name)!=expected:
                raise ValueError('frozen experiment data changed: '+name)
        with np.load(data_root/'observations.npz',allow_pickle=False) as archive:
            data={k:archive[k].copy() for k in archive.files}
        if not all(np.isfinite(data[k]).all() for k in ('vertices','coefficients','times','conditions')):
            raise ValueError('nonfinite observations')
        pca=ShapePCA.load(data_root/'pca.npz')
        groups=patient_indices(data['pids'],data['splits'])
        train=[(pid,indices) for pid,indices in groups.items()
               if data['splits'][indices[0]]=='train' and len(np.unique(data['times'][indices]))>1]
        if len(train)!=config['train_temporal_subjects']:
            raise ValueError('training temporal subject count mismatch')
        seed=config['seed']
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.set_num_threads(config['cpu_threads'])
        if not torch.cuda.is_available():
            raise ValueError('configured GPU is unavailable')
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        device=torch.device('cuda:0')
        model=ODEFuncWithAttention(150,1).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],
            betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=config['scheduler_step_epochs'],
                                                 gamma=config['scheduler_gamma'])
        mean=torch.tensor(pca.mean_,dtype=torch.float32,device=device)
        components=torch.tensor(pca.components_,dtype=torch.float32,device=device)
        tensors={pid:(torch.tensor(data['coefficients'][indices],dtype=torch.float32,device=device),
                      torch.tensor(data['times'][indices],dtype=torch.float32,device=device),
                      torch.tensor(data['conditions'][indices],dtype=torch.float32,device=device)) for pid,indices in train}
        atomic_json(report/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,
            device=str(device),gpu_name=torch.cuda.get_device_name(device),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            parameter_count=sum(p.numel() for p in model.parameters()),
            effective_attention='one independent patient state per solve; singleton attention softmax equals one, as in public train.py'))
        update(status='training',train_temporal_subjects=len(train),observations=len(data['vertices']))
        log_fields=['epoch','train_patient_mean_squared_pca_l2','ordered_pairs','optimizer_steps','learning_rate','epoch_seconds']
        with (report/'epochs.csv').open('w',newline='',buffering=1) as file:
            writer=csv.DictWriter(file,fieldnames=log_fields);writer.writeheader()
            for epoch in range(1,config['epochs']+1):
                epoch_start=time.monotonic()
                model.train();order=list(tensors);random.shuffle(order)
                losses=[];epoch_losses=[];pairs=0;steps=0
                lr=optimizer.param_groups[0]['lr']
                for pid in order:
                    coefficients,times,conditions=tensors[pid]
                    factor=random.uniform(*config['augmentation_scale'])
                    shapes=scale_pca_shapes(coefficients,mean,components,factor)
                    loss,count=trajectory_loss(model,shapes,times,conditions)
                    if loss is None or not torch.isfinite(loss):
                        raise ValueError('missing or nonfinite training loss: '+pid)
                    epoch_losses.append(float(loss.detach()));losses.append(loss);pairs+=count
                    if len(losses)==config['batch_subjects']:
                        optimizer_batch_step(optimizer,losses);losses=[];steps+=1
                if losses:
                    optimizer_batch_step(optimizer,losses);steps+=1
                scheduler.step()
                record=dict(epoch=epoch,train_patient_mean_squared_pca_l2=float(np.mean(epoch_losses)),
                    ordered_pairs=pairs,optimizer_steps=steps,learning_rate=lr,
                    epoch_seconds=time.monotonic()-epoch_start)
                writer.writerow(record);file.flush()
                print(json.dumps(record),flush=True)
                if epoch==1 or epoch%config['checkpoint_every_epochs']==0 or epoch==config['epochs']:
                    temporary=output/'latest.pt.tmp'
                    torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),config=config,config_sha256=digest(config_path),
                        random_state=random.getstate(),numpy_state=np.random.get_state(),
                        torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all()),temporary)
                    temporary.replace(output/'latest.pt')
                update(epochs_completed=epoch,last_epoch=record,
                    estimated_remaining_seconds=(time.monotonic()-started)/epoch*(config['epochs']-epoch))
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite final model')
        update(status='evaluating_final_epoch')
        result=evaluate(model,data,pca,groups,device,output,report)
        if result['metrics']['all']['targets']!=config['test_future_targets']:
            raise ValueError('held-out future evaluation coverage mismatch')
        update(status='completed_observed_baseline',evaluation=result,
            checkpoint_sha256=digest(output/'latest.pt'),estimated_remaining_seconds=0)
        print(json.dumps(status,indent=2),flush=True)
    except BaseException as error:
        update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
