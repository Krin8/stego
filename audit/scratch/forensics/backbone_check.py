from common import *
inv=inventory();reference=torch.load(inv['seed0']['path'],map_location='cpu',mmap=True,weights_only=False)['state_dict'];results={}
keys=[k for k in reference if k.startswith('net.model.')]
for name,item in inv.items():
 if not item['exists']:continue
 d=torch.load(item['path'],map_location='cpu',mmap=True,weights_only=False)['state_dict']
 results[name]=dict(visual_trunk_tensors=len(keys),exactly_equal_to_seed0=all(torch.equal(reference[k],d[k]) for k in keys));del d
assert all(r['exactly_equal_to_seed0'] for r in results.values())
dump('backbone_check.json',dict(script=__file__,results=results,elapsed_seconds=elapsed()));print(results,elapsed())
