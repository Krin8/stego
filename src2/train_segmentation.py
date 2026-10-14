"""Fresh seed-0 head training: P1 or feature-kNN; final checkpoint only."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import fcntl
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import torch
from src2.utils import parser,load_config,setup,resolve_path,new_output,write_json,sha256,labels_closed
from src2.data import FeatureStore
from src2.modules import StegoHead,objective
from src2.correspondence import prepare_correspondence
from src2.state import capture
from src2.collapse_monitor import measure


def overlap_check():
    result=subprocess.run(['ps','-axo','pid,ppid,state,etime,command'],text=True,capture_output=True,check=True)
    hits=[]
    for line in result.stdout.splitlines()[1:]:
        parts=line.strip().split(None,4)
        if len(parts)!=5:continue
        pid=int(parts[0]);cmd=parts[4]
        if pid in (os.getpid(),os.getppid()):continue
        # macOS can exec Python in the launcher PID and leave caffeinate as its
        # child. That helper repeats our command in argv but is not a trainer.
        if int(parts[1])==os.getpid() and Path(cmd.split()[0]).name=='caffeinate':continue
        if 'python' in cmd.lower() and any(s in cmd for s in ('train_segmentation','/train.py','resume_seed','run_suite.py','train_fcn','train_gpu_end_to_end','train_cnn_head')):
            hits.append(line)
    if hits:raise RuntimeError('Other training processes: '+'; '.join(hits))
    return result.stdout


def train(cfg, output):
    setup(cfg)
    process_snapshot=overlap_check()  # Failure to inspect processes is a hard stop.
    with labels_closed():
        store=FeatureStore(cfg);ids=store.indices('train')
        c=SimpleNamespace(**cfg['training'])
        if c.optimizer!='Adam' or c.lr_schedule is not None or c.batch_size!=16:
            raise ValueError('This trainer implements the original Adam/no-scheduler/batch-16 protocol')
        out=new_output(output)
        write_json(out/'resolved_config.json',cfg)
        (out/'process_snapshot.txt').write_text(process_snapshot)
        # A shared package lock catches concurrent src2 launch attempts.
        lock=(Path(__file__).resolve().parent/'.training.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def batch(rows):return store.batch([ids[int(j)] for j in rows])
        model=StegoHead()
        criterion,partners,pair_metadata,neighbors=prepare_correspondence(cfg,store,c)
        if neighbors is not None:
            np.save(out/'train_neighbors.npy',neighbors,allow_pickle=False)
            pair_metadata['array_sha256']=sha256(out/'train_neighbors.npy')
        write_json(out/'correspondence.json',pair_metadata)
        optimizer=torch.optim.Adam(model.parameters(),lr=c.lr);rng=np.random.default_rng(0)
        mrng=np.random.default_rng(c.monitor_seed);patients=sorted({store.records[i]['patient'] for i in ids},key=int)
        monitor=[int(mrng.choice([j for j,i in enumerate(ids) if store.records[i]['patient']==p])) for p in patients]
        positive,valid=partners(monitor,mrng);fixed=batch(monitor),batch(positive)
        initial=measure(model,criterion,c,fixed,valid);write_json(out/'initial.json',initial)
        order=rng.permutation(len(ids));cursor=0;bad=0;rows=[];start=time.monotonic()
        with (out/'train.jsonl').open('x') as log:
            for step in range(1,c.batches+1):
                if cursor+c.batch_size>len(order):order=rng.permutation(len(ids));cursor=0
                selected=order[cursor:cursor+c.batch_size];cursor+=c.batch_size
                positive,batch_valid=partners(selected,rng)
                if batch_valid is not None:criterion.valid=batch_valid
                model.train();f,z=model(batch(selected));fp,zp=model(batch(positive))
                loss,parts=objective(criterion,c,f,fp,z,zp)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                row=dict(seed=0,batch=step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()})
                if step==10:
                    projected=(time.monotonic()-start)/10*c.batches
                    write_json(out/'timing.json',dict(projected_seconds=projected))
                    if projected>5400:raise RuntimeError('Projected over 90 minutes; stop and review budget')
                if step%c.monitor_every==0:
                    values=measure(model,criterion,c,fixed,valid);bad=bad+1 if not values['healthy'] else 0
                    row.update(**values,consecutive_bad_checks=bad,collapse_detected=bad>=3);rows.append(row)
                    print(row,flush=True)
                import json
                log.write(json.dumps(row)+'\n');log.flush()
        state=capture(model,optimizer,rng,order,cursor,step,bad)
        state.update(seed=0,config=cfg,source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
        torch.save(state,out/'final.pt')
        write_json(out/'complete.json',dict(batches=step,initial=initial,monitors=rows,
            checkpoint_sha256=sha256(out/'final.pt'),labels_read=False,resumed=False))
        lock.close()
        print(out/'final.pt')


def main():
    p=parser('train_config.yaml',__doc__);p.add_argument('--output')
    a=p.parse_args();cfg=load_config(a.config)
    train(cfg,a.output or resolve_path(cfg,cfg['output']))


if __name__=='__main__':main()
