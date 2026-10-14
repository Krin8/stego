"""Actual saved model inference on twenty fixed validation slices, no fitting."""
from diag_common import *
from modules import BiomedCLIPFeaturizer,ClusterLookup
import torch.nn.functional as F
plan=json.loads((OUT/'plan.json').read_text());ix=plan['collapse_indices'];assert len(ix)==20
ckpath=ROOT/'checkpoints/chaos/chaos_chaos_v2_ct_date_Sep30_15-52-07/checkpoint-epoch=10-step=3600-mIoU=test/cluster/mIoU=46.1792.ckpt'
ck=torch.load(ckpath,map_location='cpu',mmap=True,weights_only=False);sd=ck['state_dict']
net=BiomedCLIPFeaturizer(70,cfg()).cpu().eval()
net.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('net.')},strict=True)
probe=ClusterLookup(70,2);probe.load_state_dict({k[len('cluster_probe.'):]:v for k,v in sd.items() if k.startswith('cluster_probe.')},strict=True)
images=np.load(OUT/'val_images.npy',mmap_mode='r');codes=[];features=[];counts=torch.zeros(2,dtype=torch.int64);each=[]
with torch.no_grad():
 for start in range(0,20,4):
    indices=ix[start:start+4];f,c=net(torch.from_numpy(np.array(images[indices])))
    features.append(f);codes.append(c)
    _,assign=probe(F.interpolate(c,(224,224),mode='bilinear',align_corners=False),None)
    for i,z in zip(indices,assign.argmax(1)):
        v=torch.bincount(z.flatten(),minlength=2);counts+=v
        each.append(dict(sample=plan['val'][i]['stem'],patient=plan['val'][i]['patient_id'],cluster_counts=v.tolist()))
 c=torch.cat(codes);f=torch.cat(features);cn=F.normalize(c,dim=1)
 means=cn.flatten(2).mean(2);pair_cos=means@means.T;off=~torch.eye(20,dtype=torch.bool)
 patients=[plan['val'][i]['patient_id'] for i in ix]
 different=torch.tensor([[a!=b for b in patients] for a in patients])
 def summary_std(v):
    global_std=v.permute(1,0,2,3).reshape(v.shape[1],-1).std(dim=1,unbiased=False)
    within=v.flatten(2).std(dim=2,unbiased=False)
    return dict(global_per_channel_mean=float(global_std.mean()),global_per_channel_min=float(global_std.min()),global_per_channel_max=float(global_std.max()),within_image_per_channel_mean=float(within.mean()))
 result=dict(script=__file__,checkpoint=str(ckpath.relative_to(ROOT)),checkpoint_global_step=int(ck['global_step']),slice_count=20,patient_count=len(set(patients)),patients=sorted(set(patients),key=int),requirement_ten_patients='UNVERIFIED / impossible in current CT validation split: only four exist',cluster_usage=counts.tolist(),cluster_fractions=(counts/counts.sum()).tolist(),cross_image_code_cosine=float(pair_cos[off].mean()),cross_patient_code_cosine=float(pair_cos[different].mean()),raw_code_pixel_std=summary_std(c),unit_code_pixel_std=summary_std(cn),code_norm_mean=float(c.norm(dim=1).mean()),finite=bool(torch.isfinite(c).all()),per_slice=each,elapsed_seconds=elapsed())
dump('collapse.json',result);print(json.dumps(result,indent=2))
with (ROOT/'audit/DIAG.md').open('a') as out:out.write('\n## Checkpoint collapse\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
