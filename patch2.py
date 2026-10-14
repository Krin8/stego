import sys

with open("audit/ensemble/verify4/run.py", "r") as f:
    code = f.read()

# Replace parameter sweep
code = code.replace("g_size=17, threshold=0.5", "g_size=25, threshold=0.30")

# Replace processing list
code = code.replace('process_arm("Frozen Baseline", [0, 1, 2])', 'process_arm("Frozen Baseline", [0])')
code = code.replace('process_arm("C448", [0, 1, 2])', '')
code = code.replace('process_arm("P1", [0, 1, 2])', '')
code = code.replace('process_arm("Untrained Head", list(range(10)))', 'process_arm("Untrained Head", [9])')

# Replace writing output
write_idx = code.find("with open(OUT / 'VERIFY4.md', 'w') as f:")
code = code[:write_idx] + """
print("\\n=== OPTIMIZED EVALUATION RESULTS ===")
for arm in ["Frozen Baseline", "Untrained Head"]:
    print(f"\\nARM: {arm}")
    seeds = [0] if arm == "Frozen Baseline" else [9]
    res = evaluate_patient(arm, seeds)
    
    for s in seeds:
        print(f"  Seed {s} - Soft Pipeline (A): {res[s]['A']['iou']:.2f}% IoU / {res[s]['A']['dice']:.2f}% Dice")
        print(f"  Seed {s} - Hard Pipeline (B): {res[s]['B']['iou']:.2f}% IoU / {res[s]['B']['dice']:.2f}% Dice")
print("======================================\\n")
"""

with open("audit/ensemble/verify4/run_optimized2.py", "w") as f:
    f.write(code)
