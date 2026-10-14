"""Metadata inventory of every file; static scan of project-owned Python only."""
import ast,collections,csv,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
rows=[];code=[];errors=[];duplicates=collections.defaultdict(list)
def category(parts,suffix):
    if '.git' in parts:return 'git-internal'
    if any(p.startswith('.venv') or p in ['site-packages','node_modules'] for p in parts):return 'environment/dependency'
    if any(p in ['__pycache__','.cache','mpl','torchinductor_navneetbavineni'] for p in parts):return 'generated-runtime-cache'
    if parts[0]=='Dataset' or 'pytorch_data_dir' in parts:return 'dataset/derivative'
    if suffix in ['.pt','.pth','.ckpt','.safetensors','.bin']:return 'weights/checkpoint'
    if suffix in ['.npy','.npz']:return 'feature/prediction/metric-array'
    if suffix=='.py':return 'project-python'
    if suffix in ['.md','.rst','.pdf']:return 'report/document'
    if suffix in ['.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.sha256']:return 'config/metadata/results'
    if suffix in ['.png','.jpg','.jpeg','.svg']:return 'figure/image'
    if suffix in ['.log','.txt','.out']:return 'log/text'
    return 'other'
for directory,dirs,files in os.walk(ROOT,followlinks=False):
    if Path(directory).is_relative_to(OUT):dirs[:]=[];continue
    for filename in files:
        p=Path(directory)/filename;rel=p.relative_to(ROOT);st=p.lstat();cat=category(rel.parts,p.suffix.lower())
        row=dict(path=str(rel),bytes=st.st_size,role=cat,coverage='metadata only',symlink=p.is_symlink())
        if cat=='project-python' and not p.is_symlink():
            raw=p.read_bytes();row['coverage']='full-file static AST scan';digest=hashlib.sha256(raw).hexdigest();duplicates[digest].append(str(rel))
            try:
                t=ast.parse(raw,filename=str(rel));calls=[]
                for n in t.body:
                    if isinstance(n,(ast.Expr,ast.Assign,ast.AnnAssign)):
                        for c in ast.walk(n):
                            if isinstance(c,ast.Call):calls.append(dict(line=c.lineno,call=ast.unparse(c.func)))
                code.append(dict(path=str(rel),lines=len(raw.splitlines()),sha256=digest,
                    definitions=[dict(name=n.name,line=n.lineno,kind=type(n).__name__) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))],
                    imports=[dict(line=n.lineno,module=n.module if isinstance(n,ast.ImportFrom) else ','.join(a.name for a in n.names)) for n in ast.walk(t) if isinstance(n,(ast.Import,ast.ImportFrom))],top_level_calls=calls))
            except (SyntaxError,UnicodeDecodeError) as exc:errors.append(dict(path=str(rel),error=str(exc)));row['coverage']='static parse failed'
        rows.append(row)
with (OUT/'FILES.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['path','bytes','role','coverage','symlink']);writer.writeheader();writer.writerows(rows)
summary=dict(total_files=len(rows),total_bytes=sum(r['bytes'] for r in rows),
    categories={c:dict(files=sum(r['role']==c for r in rows),bytes=sum(r['bytes'] for r in rows if r['role']==c)) for c in sorted({r['role'] for r in rows})},
    project_python_files=len(code),project_python_lines=sum(c['lines'] for c in code),syntax_errors=errors,
    identical_python_groups=[v for v in duplicates.values() if len(v)>1])
(OUT/'inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'python_index.json').write_text(json.dumps(code,indent=2)+'\n')
with (OUT/'PYTHON_FILES.md').open('w') as f:
    f.write('# Python file index\n\nAll project Python files were read and AST-scanned without executing them. This catalog is not a claim of complete semantic or runtime verification. Detailed findings concern the final-pipeline dependency paths.\n\n')
    for c in sorted(code,key=lambda x:x['path']):
        definitions=', '.join(f"{d['name']}:{d['line']}" for d in c['definitions']) or 'top-level script'
        f.write(f"- `{c['path']}` — {c['lines']} lines; {definitions}.\n")
print(json.dumps({k:v for k,v in summary.items() if k!='identical_python_groups'},indent=2))
