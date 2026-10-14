"""Read-only digest gate; no model or tensor loading occurs here."""
from pathlib import Path
import hashlib,json,datetime
H=Path(__file__).resolve().parent
specs={
 'D2':dict(folder='dino-vitb16',commit='f205d5d8e640a89a2b8ef0369670dfc37cc07fc2',expected='2c465a30aa590f9b3fbe36feb962a1eb35635da809643f42857cf64ee450007c',bin_expected='a064e36c67289caaa5c949c0b3f7f31a0fcbcba5721f5fa12419933ec1f4fe6e'),
 'D3':dict(folder='dinov2-base',commit='f9e44c814b77203eaa57a6bdbbd535f21ede1415',expected='d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841')}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
result={};bad=[]
for arm,s in specs.items():
 folder=H/'weights'/s['folder'];p=folder/'model.safetensors';actual=digest(p)
 pre=json.loads((folder/'preprocessor_config.json').read_text());config=json.loads((folder/'config.json').read_text())
 matches=actual==s['expected'];image_net=pre['image_mean']==[.485,.456,.406] and pre['image_std']==[.229,.224,.225]
 row=dict(file=str(p),bytes=p.stat().st_size,expected_sha256=s['expected'],actual_sha256=actual,digest_matches=matches,hub_commit_user_supplied=s['commit'],hub_commit_verification='User-supplied provenance, not checked remotely; no network',preprocessor=pre,model_config=config,imagenet_constants_verified=image_net,preprocessor_sha256=digest(folder/'preprocessor_config.json'),config_sha256=digest(folder/'config.json'))
 if arm=='D2':
  b=folder/'pytorch_model.bin';d=digest(b);row['source_bin']=dict(file=str(b),expected_sha256=s['bin_expected'],actual_sha256=d,matches=d==s['bin_expected'])
  if d!=s['bin_expected']:bad.append('D2 original binary digest differs')
 if not matches or not image_net:bad.append(arm+' digest or normalization differs')
 result[arm]=row
(H/'weight_verification.json').write_text(json.dumps(dict(producer='verify_weights.py',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=not bad,arms=result,errors=bad),indent=2)+'\n')
if bad:raise SystemExit('; '.join(bad))
print('PASS: both safetensors and D2 original binary digests match; both processors use ImageNet constants')
