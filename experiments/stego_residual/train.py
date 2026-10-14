"""Fresh bounded run; exclusive lock, process check, final checkpoint only."""
import os
os.environ['STEGO_STAGE']='training'
from runtime import *
from model import StegoHead,objective
from reference_loss import ContrastiveCorrelationLoss
from state import capture
import argparse,subprocess,fcntl
from sklearn.cluster import KMeans

def overlap_check():
    raw=subprocess.check_output(['ps','-axo','pid,ppid,state,etime,command'],text=True)
    hits=[]
    for line in raw.splitlines()[1:]:
        parts=line.strip().split(None,4)
        if len(parts)!=5:continue
        pid=int(parts[0]);cmd=parts[4]
        if pid in (os.getpid(),os.getppid()):continue
        if ('python' in cmd.lower() and any(x in cmd for x in ['train_segmentation','/train.py','resume_seed','run_suite.py'])):
            hits.append(line)
    return raw,hits

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
    lock=(PACKAGE/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw,hits=overlap_check()
    dump(f'logs/seed{a.seed}_overlap.json',dict(pid=os.getpid(),process_snapshot=raw,other_training=hits))
    if hits:raise RuntimeError('Other training process: '+repr(hits))
    c=read(HERE/'configs/locked.json');assert sha(HERE/'configs/locked.json')==(HERE/'configs/locked.sha256').read_text().split()[0]
    assert c['candidate']=='D448' and c['batches'] in (150,500)
    assert not (HERE/'predictions/stageA_manifest.json').exists()
    out=HERE/f'checkpoints/seed{a.seed}';out.mkdir(exist_ok=False)
    log=HERE/f'logs/seed{a.seed}_train.jsonl';assert not log.exists()
    rr=records();ids=[i for i,r in enumerate(rr) if r['split']=='train']
    from chaos_splits import load_splits,patient_split
    split=load_splits();assert all(patient_split(split,'CT',r['patient'])==r['split'] for r in rr)
    features=np.load(OLD/'B2_448_features.npy',mmap_mode='r');neighbors=np.load(TRAIN_CACHE,allow_pickle=False)
    assert neighbors.shape==(len(ids),30) and np.all(neighbors!=np.arange(len(ids))[:,None])
    def batch(rows):
        return torch.from_numpy(np.stack([features[ids[int(j)]] for j in rows])).reshape(len(rows),28,28,768).permute(0,3,1,2).contiguous()
    seed_all(a.seed);model=StegoHead();cfg=SimpleNamespace(**c);criterion=ContrastiveCorrelationLoss(cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr);rng=np.random.default_rng(a.seed)
    mrng=np.random.default_rng(c['monitor_seed']);patients=sorted({rr[i]['patient'] for i in ids},key=int)
    monitor=[int(mrng.choice([j for j,i in enumerate(ids) if rr[i]['patient']==p])) for p in patients]
    positive=[int(neighbors[j,mrng.integers(7)]) for j in monitor];fixed=batch(monitor),batch(positive)
    def measure():
        model.eval()
        with torch.random.fork_rng(),torch.no_grad():
            torch.manual_seed(c['monitor_seed']);f,z=model(fixed[0]);fp,zp=model(fixed[1]);loss,_=objective(criterion,cfg,f,fp,z,zp)
            z=F.normalize(z,dim=1);std=float(z.flatten(2).std(2).mean());means=z.flatten(2).mean(2)
            cosine=float((means@means.T)[~torch.eye(len(means),dtype=torch.bool)].mean())
            flat=z.permute(0,2,3,1).reshape(-1,768).numpy()
            km=KMeans(n_clusters=8,random_state=c['monitor_seed'],n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd').fit(flat)
            usage=np.bincount(km.labels_,minlength=8)/len(flat)
        return dict(code_std=std,cross_image_cosine=cosine,cluster_usage=usage.tolist(),
          cluster_entropy=float(-(usage[usage>0]*np.log(usage[usage>0])).sum()),monitor_loss=float(loss),healthy=bool(cosine<.99 and std>.01))
    initial=measure();dump(f'logs/seed{a.seed}_initial.json',dict(seed=a.seed,batch=0,**initial))
    order=rng.permutation(len(ids));cursor=0;bad=0;rows=[];start=time.monotonic()
    for step in range(1,c['batches']+1):
        if cursor+c['batch_size']>len(order):order=rng.permutation(len(ids));cursor=0
        selected=order[cursor:cursor+c['batch_size']];cursor+=c['batch_size']
        positive=[neighbors[j,rng.integers(7)] for j in selected]
        model.train();f,z=model(batch(selected));fp,zp=model(batch(positive));loss,parts=objective(criterion,cfg,f,fp,z,zp)
        assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        row=dict(seed=a.seed,batch=step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()})
        if step==10:
            elapsed=time.monotonic()-start;projected=elapsed/10*c['batches'];dump(f'logs/seed{a.seed}_timing.json',dict(first10_seconds=elapsed,projected_run_seconds=projected,budget=c['batches']))
            print('TIMING',elapsed,projected,flush=True)
            if projected>5400:raise RuntimeError('PROJECTED_OVER_90_MINUTES: stop and ask user')
        if step%50==0:
            m=measure();bad=bad+1 if not m['healthy'] else 0
            row.update(**m,consecutive_bad_checks=bad,collapse_detected=bad>=3);rows.append(row)
            print(json.dumps(row),flush=True)
        with log.open('a') as stream:stream.write(json.dumps(row)+'\n')
    state=capture(model,optimizer,rng,order,cursor,step,bad)
    state.update(seed=a.seed,config_sha256=sha(HERE/'configs/locked.json'),source_sha256={p.name:sha(p) for p in PACKAGE.glob('*.py')},monitor_indices=[ids[j] for j in monitor])
    torch.save(state,out/'final.pt')
    passed=all(r['healthy'] for r in rows) and rows[-1]['cross_image_cosine']<initial['cross_image_cosine']
    dump(f'logs/seed{a.seed}_complete.json',dict(seed=a.seed,batches=step,seconds=time.monotonic()-start,screen_pass=passed,
      initial=initial,monitors=rows,checkpoint_sha256=sha(out/'final.pt'),labels_opened=False,resumed=False))
    ledger(f'logs/seed{a.seed}_access.json')
if __name__=='__main__':main()
