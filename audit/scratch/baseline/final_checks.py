from common import *
import subprocess
summary=json.loads((OUT/'summary.json').read_text());assert len(summary['rows'])==17 and summary['proxy_count']==50
assert len(summary['proxy_correlations'])==4
chunks=[json.loads(p.read_text()) for p in sorted(OUT.glob('proxies_*_*.json'))]
assert len(chunks)==5 and all(len(x['results'])==10 and x['elapsed_seconds']<570 for x in chunks)
measurements=[r for chunk in chunks for r in chunk['results']];assert sorted(r['batch'] for r in measurements)==list(range(100,5001,100))
assert all(r['feature_reuse_error']<1e-5 and r['code_reuse_error']<1e-5 and r['backbone_equal'] for r in measurements)
for tag in ['best600','final']:
 r=next(r for r in measurements if r['batch']==(600 if tag=='best600' else 5000));assert r['forensics_loss_max_abs_difference']<1e-6
assert all(Path(r['checkpoint']).is_file() for r in summary['rows'] if 'checkpoint' in r)
assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).strip()
report=(ROOT/'audit/BASELINE.md').read_text();assert 'VERDICT_PLACEHOLDER' not in report and 'UNVERIFIED/incomplete' not in report
for p in OUT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
result=dict(script=__file__,status='PASS',fit_rows=12,head_rows=5,proxy_checkpoints=50,proxy_chunk_runtimes=[x['elapsed_seconds'] for x in chunks],max_feature_error=max(r['feature_reuse_error'] for r in measurements),max_code_error=max(r['code_reuse_error'] for r in measurements),best600_final_loss_reproduction_errors={str(r['batch']):r['forensics_loss_max_abs_difference'] for r in measurements if 'forensics_loss_max_abs_difference' in r},tracked_source_changes=False,elapsed_seconds=elapsed())
dump('final_checks.json',result);note('Final verification: `audit/scratch/baseline/final_checks.py` → `final_checks.json`; '+json.dumps(result));print(json.dumps(result,indent=2))
