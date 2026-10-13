"""Verify an interrupted optimizer update stream equals uninterrupted training."""
import csv
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from resume_utils import restore_checkpoint,truncate_epoch_log
from pipeline import digest,write_json


def verify(report):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed=92
    def reset():
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    def make():
        model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(8,1)).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.5)
        generator=torch.Generator().manual_seed(seed)
        return model,optimizer,scheduler,generator
    x=torch.arange(24,dtype=torch.float32,device='cuda').reshape(6,4)/24
    def epoch(model,optimizer,scheduler,generator):
        ids=torch.randperm(6,generator=generator).tolist()
        for i in ids:
            factor=random.random()+np.random.random()+float(torch.rand(()))
            noise=torch.rand(4,device='cuda')*.01
            optimizer.zero_grad();loss=((model(x[i]+noise)-factor)**2).mean();loss.backward();optimizer.step()
        scheduler.step()
    with tempfile.TemporaryDirectory(prefix='lv_resume_',dir='/data5/take/brainode/functional_tests') as temporary:
        root=Path(temporary);config=root/'config.json';write_json(config,dict(test='RNG and optimizer continuation'))
        reset();model,opt,sched,generator=make();epoch(model,opt,sched,generator)
        torch.save(dict(epoch=1,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            config_sha256=digest(config),random_state=random.getstate(),numpy_state=np.random.get_state(),
            torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),loader_rng_state=generator.get_state()),root/'latest.pt')
        epoch(model,opt,sched,generator);expected=[p.detach().clone() for p in model.parameters()]
        model,opt,sched,generator=make()
        assert restore_checkpoint(root,config,model,opt,sched,torch.device('cuda'),generator)==1
        epoch(model,opt,sched,generator)
        for a,b in zip(expected,model.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
        config.write_text('{}')
        try: restore_checkpoint(root,config,model,opt,sched,torch.device('cuda'),generator)
        except ValueError: pass
        else: raise AssertionError('changed config accepted')
        log=root/'epochs.csv';log.write_text('epoch,loss\n1,2\n2,1\n');truncate_epoch_log(log,1)
        assert log.read_text().splitlines()==['epoch,loss','1,2']
    write_json(report,dict(status='passed',bitwise_identical_after_interrupted_epoch=True,
        restored=['model','AdamW','StepLR','Python RNG','NumPy RNG','Torch CPU RNG','CUDA RNG','shuffle generator'],
        changed_config_rejected=True,post_checkpoint_log_updates_truncated=True))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    verify(parser.parse_args().report)
