"""Continue frozen evaluation automatically after both training jobs complete."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT/'artifacts/training_runs/adni1_continuous_v1'


def write(value):
    value['updated_utc'] = datetime.now(timezone.utc).isoformat()
    tmp = REPORT/'pipeline_progress.json.tmp'
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(REPORT/'pipeline_progress.json')


def main():
    status = dict(status='waiting_for_training', pid=os.getpid())
    write(status)
    while True:
        stages = {stage: json.loads((REPORT/stage/'progress.json').read_text()) for stage in ('cnn', 'ode')}
        if any(s['status'] == 'failed' for s in stages.values()):
            raise RuntimeError('training stage failed')
        if all(s['status'] == 'completed' for s in stages.values()):
            break
        time.sleep(10)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='4', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    commands = [
        ('cognition_review', ['audit/plot_cognition_results.py']),
        ('algorithm2', ['train_continuous.py', 'evaluate', '--config', str(REPORT/'deterministic_pool_config.json')]),
        ('reference_verification', ['audit/verify_algorithm2.py']),
        ('matched_controls', ['audit/matched_progressive_controls.py']),
        ('report', ['audit/summarize_continuous.py']),
    ]
    for stage, command in commands:
        status.update(status=stage); write(status)
        with (REPORT/f'{stage}.log').open('w') as file:
            subprocess.run([str(PROJECT/'.venv-mesh/bin/python'), '-u', *command], cwd=PROJECT,
                           env=environment, stdout=file, stderr=subprocess.STDOUT, check=True)
    status.update(status='completed'); write(status)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(dict(status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise
