"""Append amended calibration evidence without changing the earlier report."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent;root=P.parents[1]
r=json.loads((P/'amended_calibration.json').read_text())
lines=['\n### Amended Task 1 results\n',
 'Producer: `audit/colocated/amended_calibration.py` → `amended_calibration.json`; source/sample protocol hashes and annotation-access ledger are retained. The original 20% shift and losses reproduced within 1e-10. Partner values are diagnostic only. Full means are weighted by anchor count. No segmentation scores or annotation files were read.\n',
 '| Target | Shift | Actual attraction % | Partner constant / random / K8 | Full constant / random / K8 | Full K8 advantage | Batches passed | Admissible |',
 '| --- | ---: | ---: | --- | --- | ---: | ---: | --- |']
for row in r['summary']:
    def losses(term):return ' / '.join(f"{row['losses'][kind][term]:.9f}" for kind in ['constant','random','cluster'])
    lines.append(f"| {100*row['target']:.0f}% | {row['shift']:.15f} | {100*row['attraction']:.6f} | {losses('pos_inter')} | {losses('total')} | {row['advantage']:.9f} | {row['batches_passed']}/13 | {row['admissible']} |")
lines += ['\n| Batch (slices) | 20% | 40% | 60% | 80% | 90% |','| --- | --- | --- | --- | --- | --- |']
for batch in range(1,14):
    rr=[x for x in r['batches'] if x['batch']==batch]
    lines.append(f"| {batch} ({rr[0]['slices']}) | "+' | '.join('PASS' if x['passed'] else 'FAIL' for x in rr)+' |')
lines += [f"\n**Selected P1 target: {100*r['selected_target']:.0f}%; shift {r['selected_shift']!r}. Selection used only label-free full losses.** Self/random stay 0.794418/0.797651; C448 partner stays 0.785981. This passes the amended gate; the original partner-only failure remains recorded above.\n",
 'Preparation reuses the prior bounded runner in a new output package, with only P1 partner dispatch, candidate identity and requested config metadata changed. The copied Stage-A entry gate is generalized from D448 to its configured arm. No previous experiment is modified. `runner_reuse.json` records source copies. Optimizer/RNG serialization is verified by `experiments/stego_colocated/check_state.py` on P1 synthetic tensors: exact next-update equality for model, Adam and RNG state, without dataset reads. Production runs cannot resume; interrupted runs require a fresh attempt.\n']
text='\n'.join(lines)
(P/'amended_summary.md').write_text(text)
report=root/'audit/FIXES6.md';existing=report.read_text();assert '### Amended Task 1 results' not in existing
report.write_text(existing+text)
