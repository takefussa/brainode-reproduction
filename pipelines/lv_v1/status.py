"""Print pipeline state without stopping running workers."""
import argparse
import json
from pathlib import Path
import os

def main():
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);args=p.parse_args()
    root=args.report
    state=json.loads((root/'progress.json').read_text())
    try: os.kill(state['pid'],0);state['supervisor_alive']=True
    except ProcessLookupError: state['supervisor_alive']=False
    print(json.dumps(state,indent=2,ensure_ascii=False))
    path=root/'mesh_progress.json'
    if path.exists():
        mesh=json.loads(path.read_text())
        print(json.dumps({key:value for key,value in mesh.items() if key not in ('active_workers','worker_pids')},indent=2))
    for path in sorted((root/'stage_status').glob('*.json')):
        print(path.stem,json.loads(path.read_text())['status'])
    for path in sorted(root.glob('*/**/progress.json')):
        item=json.loads(path.read_text())
        print(str(path.relative_to(root)),item.get('status'),item.get('epochs_completed',''),item.get('steps_completed',''))

if __name__=='__main__':main()
