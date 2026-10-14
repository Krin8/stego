"""Extract only provenance/config/metric fields from saved W&B records."""
import decision_guard as g
import json,ast,re,yaml
rows=['## m6 — Saved-run evidence','Command: `python3 -B audit/scratch/decision_runs.py`. Metrics below are saved records, not recomputed scores. Config dim is the declared value; text mode can override it, so checkpoint shape is required to prove the effective dimension.','Run / source | Script + recorded arguments | Git commit | Config: architecture; extra clusters; dim; text; max steps; data | Last logged step/epoch; metrics','--- | --- | --- | --- | ---']
count=0; outputs=[]
for directory in sorted((g.ROOT/'logs/wandb').glob('run-*/files')):
    def readjson(name):
        p=directory/name
        return json.loads(p.read_text()) if p.exists() else {}
    meta=readjson('wandb-metadata.json'); summary=readjson('wandb-summary.json')
    cfgpath=directory/'config.yaml'; cfg={}
    if cfgpath.exists():
        raw=yaml.safe_load(cfgpath.read_text()) or {}
        value=raw.get('cfg',{}).get('value',{})
        try:cfg=ast.literal_eval(value) if isinstance(value,str) else value
        except (ValueError,SyntaxError):cfg={}
        if not cfg:cfg={k:v.get('value') for k,v in raw.items() if isinstance(v,dict) and 'value' in v}
    if not (meta or summary or cfg):continue
    count+=1; run=directory.parent.name
    info={k:cfg.get(k,'UNVERIFIED') for k in ['arch','extra_clusters','dim','use_text_prompts','max_steps','pytorch_data_dir','chaos_modality','use_preprocessed_data']}
    metrics={k:v for k,v in summary.items() if 'mIoU' in k or 'Accuracy' in k or k in ['trainer/global_step','epoch']}
    command=dict(executable=meta.get('executable','UNVERIFIED'),script=meta.get('program',meta.get('codePath','UNVERIFIED')),args=meta.get('args','UNVERIFIED: absent'))
    rel=str(directory.relative_to(g.ROOT))
    rows.append(f'`{rel}` (`wandb-metadata.json`, `config.yaml`, `wandb-summary.json`) | `{command}` | `{meta.get("git",{}).get("commit","UNVERIFIED")}` | `{info}` | `{metrics}`')
    out=directory/'output.log'
    if out.exists():
        hits=[]
        with out.open(errors='replace') as f:
            for number,line in enumerate(f,1):
                if any(t in line.lower() for t in ['checkpoint','python ','loaded knn','max_steps','error','traceback']):
                    hits.append(f'`{out.relative_to(g.ROOT)}:{number}` {line.strip()[:400]}')
        outputs.extend(hits[-5:])
g.append('\n'.join(rows)+f'\n\nSaved run records found: {count}.\n\n### Selected output-log provenance lines\n\n'+'\n\n'.join(outputs))
print(dict(runs=count,output_log_hits=len(outputs)))
