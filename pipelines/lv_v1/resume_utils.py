"""Resume locally generated, config-verified epochs including all RNG streams."""
import csv
import hashlib
from pathlib import Path
import random
import numpy as np
import torch


def restore_checkpoint(output, config_path, model, optimizer, scheduler, device, generator=None):
    checkpoint = Path(output)/'latest.pt'
    if not checkpoint.exists():
        if any(p.name != 'latest.pt.tmp' for p in Path(output).iterdir()):
            raise ValueError('existing training output has no verified checkpoint')
        return 0
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved['config_sha256'] != hashlib.sha256(Path(config_path).read_bytes()).hexdigest():
        raise ValueError('resume checkpoint configuration changed')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    random.setstate(saved['random_state'])
    np.random.set_state(saved['numpy_state'])
    torch.set_rng_state(saved['torch_rng_state'].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng_state']])
    if generator is not None:
        if saved.get('loader_rng_state') is None:
            raise ValueError('CNN checkpoint lacks shuffle generator state')
        generator.set_state(saved['loader_rng_state'].cpu())
    return int(saved['epoch'])


def truncate_epoch_log(path, epoch):
    """Discard logged updates newer than the last atomic checkpoint."""
    if not epoch:
        return
    path = Path(path)
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row['epoch']) <= epoch]
    if [int(row['epoch']) for row in rows] != list(range(1, epoch+1)):
        raise ValueError('resume log does not cover the checkpoint epochs')
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)
