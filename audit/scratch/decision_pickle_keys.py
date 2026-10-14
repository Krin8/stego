"""Inspect only uncompressed pickle opcodes in a damaged ZIP's first entry."""
import decision_guard as g
import struct,pickletools,json
path=g.ROOT/'models/model_epoch_0720_iter_085000.pth'
with path.open('rb') as f:
    header=f.read(30);parts=struct.unpack('<4s5H3I2H',header)
    assert parts[0]==b'PK\x03\x04' and parts[3]==0,'Unsupported archive header/compression'
    name=f.read(parts[-2]);f.read(parts[-1]);start=f.tell()
    assert name.endswith(b'data.pkl')
    strings=[];stop=False;ops=0
    for op,arg,pos in pickletools.genops(f):
        ops+=1
        if f.tell()-start>2_000_000:raise RuntimeError('Metadata exceeded bounded read')
        if isinstance(arg,str):strings.append(arg)
        if op.name=='STOP':stop=True
    result=dict(path=str(path),entry=name.decode(),metadata_bytes_read=f.tell()-start,opcodes=ops,pickle_stop=stop,key_candidates=[x for x in strings if any(t in x for t in ['cluster_probe','linear_probe','cluster1','cluster2','patch_embed','pos_embed','cls_token','teacher','student','backbone'])],first_strings=strings[:35],note='Static opcode inspection only; no pickle execution or tensor storage read. Archive remains unloadable by torch.load; string candidates alone do not reconstruct its full state dictionary.')
g.append('## Damaged archive: static metadata inspection\n\nCommand: `python3 -B audit/scratch/decision_pickle_keys.py`. No deserialization, tensor loading or archive repair.\n\n```json\n'+json.dumps(result,indent=2)+'\n```')
print(json.dumps(result))
