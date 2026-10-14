import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'final_pipeline'

results = {}
for seed in [0, 1, 2]:
    old_meta = json.loads((OLD / f'all20_seed{seed}.json').read_text())
    new_meta = json.loads((HERE / f'all20_seed{seed}.json').read_text())
    
    old_c = old_meta['liver_clusters'][0]
    new_c = new_meta['liver_cluster']
    
    results[f'seed_{seed}'] = {
        'label_selected': old_c,
        'N3_selected': new_c,
        'match': old_c == new_c,
        'abstention': new_meta.get('abstention_status', False)
    }

results['all_match'] = all(r['match'] for k, r in results.items() if k.startswith('seed_'))
results['any_abstained'] = any(r['abstention'] for k, r in results.items() if k.startswith('seed_'))

(HERE / 'naming_verification.json').write_text(json.dumps(results, indent=2))
print("Verification complete:", results)
