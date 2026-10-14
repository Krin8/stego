"""Original P1 train-only monitor definitions; monitor IDs are not liver IDs."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .modules import objective


def measure(model, criterion, cfg, fixed, valid):
    model.eval()
    if valid is not None: criterion.valid = valid
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(cfg.monitor_seed)
        f,z=model(fixed[0]);fp,zp=model(fixed[1]);loss,_=objective(criterion,cfg,f,fp,z,zp)
        z=F.normalize(z,dim=1);std=float(z.flatten(2).std(2).mean());means=z.flatten(2).mean(2)
        cosine=float((means@means.T)[~torch.eye(len(means),dtype=torch.bool)].mean())
        flat=z.permute(0,2,3,1).reshape(-1,70).numpy()
        km=KMeans(n_clusters=8,random_state=cfg.monitor_seed,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd').fit(flat)
        usage=np.bincount(km.labels_,minlength=8)/len(flat)
    return dict(code_std=std,cross_image_cosine=cosine,cluster_usage=usage.tolist(),
        cluster_entropy=float(-(usage[usage>0]*np.log(usage[usage>0])).sum()),
        monitor_loss=float(loss),healthy=bool(cosine<.99 and std>.01))
