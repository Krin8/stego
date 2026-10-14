"""Run the real BiomedCLIPFeaturizer; capture five normalized blocks in its actual call."""
from diag_common import *
from modules import BiomedCLIPFeaturizer
from numpy.lib.format import open_memmap
import argparse
p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val'],required=True);args=p.parse_args()
plan=json.loads((OUT/'plan.json').read_text());rows=plan[args.split];ds=dataset(args.split)
net=BiomedCLIPFeaturizer(70,cfg()).cpu().eval()
assert len(net.model.blocks)==12
original=net.model.get_intermediate_layers
captured={}
def capture(x,*a,**kw):
    assert kw==dict(n=1,reshape=True,norm=True),kw
    layers=original(x,n=[7,8,9,10,11],reshape=True,norm=True)
    for block,tensor in zip(range(8,13),layers):captured[block]=tensor
    return (layers[-1],)
net.model.get_intermediate_layers=capture
arrays={block:open_memmap(OUT/f'{args.split}_block{block}.npy',mode='w+',dtype='float32',shape=(len(rows),196,768)) for block in range(8,13)}
labels=open_memmap(OUT/f'{args.split}_labels.npy',mode='w+',dtype='int16',shape=(len(rows),224,224))
images=open_memmap(OUT/f'{args.split}_images.npy',mode='w+',dtype='float32',shape=(len(rows),3,224,224))
# Labels are stored for isolated supervised fitting/scoring, never used in extraction.
with torch.no_grad():
    for start in range(0,len(rows),4):
        batch=[ds[r['index']] for r in rows[start:start+4]]
        x=torch.stack([b[0] for b in batch]);f,c=net(x)
        assert f.shape[1:]==(768,14,14) and torch.equal(f,captured[12])
        for block,t in captured.items():
            assert t.shape[1:]==(768,14,14) and torch.isfinite(t).all()
            arrays[block][start:start+len(batch)]=t.flatten(2).transpose(1,2).numpy()
        labels[start:start+len(batch)]=torch.stack([b[1] for b in batch]).numpy()
        images[start:start+len(batch)]=x.numpy()
        if start%40==0:print(args.split,start,'elapsed',elapsed(),flush=True)
for x in [*arrays.values(),labels,images]:x.flush()
report=dict(script=__file__,split=args.split,count=len(rows),blocks=list(arrays),shape=[len(rows),196,768],cpu=True,layernorm=True,cls_excluded=True,feature_dtype='float32',runtime_capture='actual extractor call instrumented to request blocks8..12 from its own timm trunk; original final output retained and equality checked',backbone_frozen=all(not p.requires_grad for p in net.full_model.parameters()),backbone_eval=not net.full_model.training,elapsed_seconds=elapsed())
dump('extract_'+args.split+'.json',report)
with (ROOT/'audit/DIAG.md').open('a') as f:f.write('\n## Extraction '+args.split+'\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
print(json.dumps(report,indent=2))
