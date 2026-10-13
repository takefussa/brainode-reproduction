"""Resume the paused LV pipeline on another host using two idle GPUs.

Scientific snapshots and completed artifacts remain unchanged. A new runtime
config records the host/device change. The supervisor starts detached.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

REPORT=Path('/home/take/brainode/artifacts/training_runs/adni1_lv_bilateral_v1')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--gpus',help='Two physical indices, e.g. 0,1; default: choose two idle devices')
    parser.add_argument('--snapshot',type=Path,help='New immutable snapshot for documented hardware supplements')
    parser.add_argument('--overlay',type=Path,help='Separately compiled PyTorch3D runtime overlay')
    parser.add_argument('--inventory',type=Path,help='Verified inventory for the current completed mesh prefix')
    parser.add_argument('--return-to-initial-host',action='store_true',help='Resume on the initial host after a later failed migration')
    args=parser.parse_args();host=socket.gethostname()
    stop=json.loads((REPORT/'host_migration_stop.json').read_text())
    if host==stop['host'] and not args.return_to_initial_host:
        raise SystemExit('Initial host is paused; returning requires --return-to-initial-host: '+host)
    state=json.loads((REPORT/'progress.json').read_text())
    if state['status'] not in ('paused_for_host_migration','failed','waiting_for_compatible_cuda_runtime'):
        raise SystemExit('Pipeline is not paused/failed; check existing supervisor before restarting')
    original=REPORT/'config.json';config=json.loads(original.read_text())
    snapshot=args.snapshot.resolve() if args.snapshot else REPORT/'code_snapshot'
    if args.snapshot:
        metadata=json.loads((snapshot/'snapshot.json').read_text())
        original_hashes=config['snapshot_sha256']
        unchanged=['mesh_fit.py','lv_mesh_fit.py','mesh_input.py','mesh_initialization.py','mesh_cohort.py',
                   'references/BrainSubcorticalStructure/models/pointnet.py','model.py','paper_training.py','paper_cognition.py',
                   'cognition.py','evaluation.py','lab_shapes.py','deterministic_pool.py',
                   'assets/eligible.csv','assets/template_audit.json','assets/lv_left_template.npz','assets/lv_right_template.npz']
        for name in unchanged:
            if digest(snapshot/name)!=original_hashes[name]:raise SystemExit('Frozen scientific core differs: '+name)
        config['snapshot_sha256']=metadata['files_sha256']
        config['table']=str(snapshot/'assets/eligible.csv')
    for name,expected in config['snapshot_sha256'].items():
        if digest(snapshot/name)!=expected:raise SystemExit('Frozen snapshot differs: '+name)
    inventory_path=args.inventory.resolve() if args.inventory else REPORT/'migration_inventory.json'
    inventory=json.loads(inventory_path.read_text())
    for entry in inventory['manifests']:
        if digest(entry['manifest'])!=entry['sha256']:raise SystemExit('Paused manifest changed: '+entry['manifest'])
    old_environment=json.loads((REPORT/'environment.json').read_text())
    packages={name:importlib.metadata.version(name) for name in old_environment['packages']}
    if packages!=old_environment['packages']:raise SystemExit('Package versions differ; match the frozen environment before resume')
    if any(entry['environment']['python']!=sys.version for entry in inventory['manifests']):
        raise SystemExit('Python build differs from cached mesh environment; use the same shared interpreter')
    import csv
    with Path(config['table']).open() as f:rows=list(csv.DictReader(f))
    missing=[r['parcellation_path'] for r in rows if not Path(r['parcellation_path']).is_file()]
    if missing:raise SystemExit('Destination cannot read source data: '+missing[0])
    if not Path(config['heavy']).is_dir():raise SystemExit('Destination cannot read generated data5 outputs')
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices=[]
    for line in query.splitlines():
        index,uuid,memory,utilization=[part.strip() for part in line.split(',')]
        devices.append(dict(index=int(index),uuid=uuid,memory_used_mib=int(memory),utilization_percent=int(utilization)))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    busy=set(apps.splitlines())
    idle=[r for r in devices if r['uuid'] not in busy and r['memory_used_mib']<512 and r['utilization_percent']<5]
    selected=[int(x) for x in args.gpus.split(',')] if args.gpus else [r['index'] for r in idle[:2]]
    if len(selected)!=2 or len(set(selected))!=2 or not set(selected)<={r['index'] for r in idle}:
        raise SystemExit('Two idle GPUs are required; available idle indices: '+str([r['index'] for r in idle]))
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,selected))
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    if args.overlay:
        overlay=args.overlay.resolve()
        runtime=json.loads((overlay/'runtime_manifest.json').read_text())
        config['runtime_sha256']=runtime['files_sha256']
        for path,expected in runtime['files_sha256'].items():
            if digest(path)!=expected:raise SystemExit('Runtime overlay changed: '+path)
        os.environ['PYTHONPATH']=str(overlay)+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
        sys.path.insert(0,str(overlay))
        config['runtime_overlay']=str(overlay)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise SystemExit('Selected CUDA devices unavailable')
    if not args.overlay:
        import pytorch3d._C
        native=Path(pytorch3d._C.__file__).resolve()
        config['runtime_sha256']={str(native):digest(native)}
        config['runtime_compute_capabilities']=[list(torch.cuda.get_device_capability(i)) for i in range(2)]
    if args.overlay:
        if any(list(torch.cuda.get_device_capability(i))!=runtime['compute_capability'] for i in range(2)):
            raise SystemExit('CUDA devices do not match the separately built runtime')
    if any(entry['environment']['cuda']!=torch.version.cuda for entry in inventory['manifests']):
        raise SystemExit('CUDA runtime differs from cached mesh environment')
    # No scientific config has been generated yet at the paused mesh stage.
    for side in ('left','right'):
        if any((REPORT/side).glob('*_config.json')):raise SystemExit('Training stage config exists; review device migration before starting')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    config.update(gpus=selected,migration_host=host,migration_parent_config_sha256=digest(original),migration_utc=stamp,
                  scientific_protocol='same cohort/time/split and effective batch16; hardware-specific CNN microbatch4' if args.snapshot else 'original protocol')
    config.update(resume_inventory=str(inventory_path),resume_inventory_sha256=digest(inventory_path))
    path=REPORT/f'config.{host}.{stamp}.json';path.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    sys.path.insert(0,str(snapshot));from pipeline import preflight
    preflight(config)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    command=[sys.executable,str(snapshot/'pipeline.py'),'--config',str(path)]
    log_path=REPORT/f'supervisor.{host}.{stamp}.log'
    with log_path.open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=snapshot,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(pid=process.pid,host=host,physical_gpus=selected,command=command,log=str(log_path),
        config=str(path),config_sha256=digest(path),start_new_session=True,completed_meshes_at_resume=inventory['completed_meshes'],
        created_utc=datetime.now(timezone.utc).isoformat())
    (REPORT/f'launch.{host}.{stamp}.json').write_text(json.dumps(launch,indent=2)+'\n')
    (REPORT/'current_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    (REPORT/'supervisor.pid').write_text(str(process.pid)+'\n')
    print(json.dumps(launch,indent=2))


if __name__=='__main__':main()
