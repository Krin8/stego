"""Disable bytecode, network and filesystem writes except the audit report."""
import sys, os
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'audit/AUDIT.md'
tempfile.tempdir='/private/tmp'
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'audit/scratch')
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['WANDB_MODE']='disabled'
def audit(event,args):
    if event=='open':
        path,mode,flags=args
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not isinstance(path,int) and Path(path).resolve()!=REPORT:
            raise PermissionError('Read-only audit blocked write: '+str(path))
    elif event=='os.mkdir' and Path(args[0]).is_dir():
        return
    elif event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.chmod','os.utime','os.link','os.symlink','socket.connect'):
        raise PermissionError('Read-only audit blocked '+event+': '+str(args)[:120])
sys.addaudithook(audit)
def record(title,text):
    print(text,flush=True)
    with REPORT.open('a') as f:f.write('\n### '+title+'\n\n```text\n'+text+'\n```\n')

def definitions(path,names,namespace):
    """Execute unchanged selected definitions, skipping unrelated import side effects."""
    import ast
    path=ROOT/path
    tree=ast.parse(path.read_text(),filename=str(path))
    selected=[]
    for node in tree.body:
        name=getattr(node,'name',None)
        if isinstance(node,ast.Assign):
            name=getattr(node.targets[0],'id',None)
        if name in names:selected.append(node)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace
