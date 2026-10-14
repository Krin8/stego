"""Finish Stage A output accounting after its process/log is closed, before Stage B."""
import os
os.environ['NAMING_STAGE']='A'
from common import *
from datetime import datetime,timezone
assert not (OUT/'stageB_started.json').exists()
m=json.loads((OUT/'stageA_manifest.json').read_text())
# Self-reference is impossible: only these two files are excluded; all other files hashed.
files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['stageA_manifest.json','stageA_manifest.sha256']]
m['output_sha256']={str(p.relative_to(OUT)):sha(p) for p in files};m['hash_exclusions']='Only manifest itself and its detached SHA256 sidecar (self-reference). All other existing StageA files, including closed logs and library metadata, hashed.';m['final_sealed_utc']=datetime.now(timezone.utc).isoformat();m['parameters']['snapshot']='Every StageA file hashed after producer exits; manifest self-hash stored separately'
dump('stageA_manifest.json',m);digest=sha(OUT/'stageA_manifest.json');(OUT/'stageA_manifest.sha256').write_text(digest+'  stageA_manifest.json\n');print(digest)
