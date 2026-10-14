"""Fail closed on overlapping Python training; preserve process evidence."""
import os,subprocess,json,datetime
from pathlib import Path
def check(output):
 rows=[]
 for line in subprocess.check_output(['ps','-axo','pid=,ppid=,state=,comm=,args='],text=True).splitlines():
  fields=line.strip().split(None,4)
  if len(fields)!=5:continue
  pid,ppid,state,comm,args=fields;rows.append(dict(pid=int(pid),ppid=int(ppid),state=state,comm=comm,args=args))
 me=os.getpid();ancestors={me};parent=next(r['ppid'] for r in rows if r['pid']==me)
 while parent and parent not in ancestors:
  ancestors.add(parent);matches=[r for r in rows if r['pid']==parent]
  parent=matches[0]['ppid'] if matches else 0
 python=[r for r in rows if 'python' in Path(r['comm']).name.lower() and r['pid'] not in ancestors]
 # Conservative: fail on every other Python process, rather than miss an indirect runner.
 wrappers=[r for r in rows if 'caffeinate' in r['comm'] and 'train_segmentation' in r['args']]
 for row in wrappers:row['child_pids']=[r['pid'] for r in rows if r['ppid']==row['pid']]
 evidence=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),self_pid=me,other_python_processes=python,training_wrappers=wrappers,clear=not python)
 Path(output).write_text(json.dumps(evidence,indent=2)+'\n')
 if python:raise RuntimeError('Overlap check found another Python process: '+json.dumps(python))
 print('PROCESS CHECK CLEAR',json.dumps(evidence),flush=True)
 return evidence
