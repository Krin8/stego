"""Bounded weight-path metadata inventory; never reads online-only contents."""
import decision_guard as g
import os,stat,signal,time
from pathlib import Path
from datetime import datetime,timezone
bases=[g.ROOT,Path.home()/'Downloads',Path.home()/'Documents']
for drive in sorted((Path.home()/'Library/CloudStorage').glob('GoogleDrive-*')):
    for child in (drive/'My Drive').iterdir():
        if any(tag in child.name.lower() for tag in ['stego','chaos','eagle','checkpoint']):bases.append(child)
bases += [p for p in Path('/Volumes').iterdir() if not p.is_symlink() and 'Google' in p.name]
found={}; failures=[]; start=time.monotonic()
def expired(*args):raise RuntimeError('bounded directory traversal reached time limit')
signal.signal(signal.SIGALRM,expired)
def err(e):failures.append(str(e))
completed=[]
try:
    for base in bases:
        for directory,dirs,files in os.walk(base,onerror=err,followlinks=False):
            if time.monotonic()-start>85:raise RuntimeError('85-second traversal budget reached')
            dirs[:]=[d for d in dirs if d not in ('.git','node_modules','.Trash','venv','.venv','site-packages','__pycache__')]
            for name in files:
                if Path(name).suffix.lower() not in ('.ckpt','.pt','.pth','.safetensors'):continue
                p=Path(directory)/name
                try:
                    s=p.stat(); found[str(p)]=(s.st_size,datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),bool(s.st_flags & stat.SF_DATALESS))
                except OSError as e:failures.append(str(e))
        completed.append(str(base))
except (RuntimeError,KeyboardInterrupt) as e:failures.append(str(e))
signal.alarm(10)
rows=['## m6 — Weight-path inventory','Command: `python3 -B audit/scratch/decision_inventory.py` (no tensor reads).',f'Completed roots: `{completed}`. Found {len(found)} files. Elapsed {time.monotonic()-start:.2f}s.',f'Incomplete/inaccessible paths: `{failures}`.','Path | Bytes | Modified UTC | Local contents','--- | ---: | --- | ---']
for path,(size,date,offline) in sorted(found.items()):rows.append(f'`{path}` | {size} | {date} | '+('ONLINE-ONLY; keys UNVERIFIED' if offline else 'available; classification pending'))
rows.insert(3,'Scope note: cloud traversal targeted first-level STEGO/CHAOS/EAGLE/checkpoint directories. Dependency environments and unrelated cloud trees were excluded after a broad traversal stalled; unsearched trees remain UNVERIFIED.')
g.append('\n'.join(rows))
print(dict(files=len(found),online_only=sum(x[2] for x in found.values()),completed=completed,errors=failures,elapsed=round(time.monotonic()-start,2)))
