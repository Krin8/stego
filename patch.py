with open("audit/ensemble/verify4/run_optimized.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == "with open(OUT / 'VERIFY4.md', 'w') as f:":
        break
    new_lines.append(line)

new_lines.append("""
print("\\n=== OPTIMIZED EVALUATION RESULTS ===")
for arm in ["Frozen Baseline", "Untrained Head"]:
    print(f"\\nARM: {arm}")
    seeds = [0] if arm == "Frozen Baseline" else [9]
    res = evaluate_patient(arm, seeds)
    
    for s in seeds:
        print(f"  Seed {s} - Soft Pipeline (A): {res[s]['A']['iou']:.2f}% IoU / {res[s]['A']['dice']:.2f}% Dice")
        print(f"  Seed {s} - Hard Pipeline (B): {res[s]['B']['iou']:.2f}% IoU / {res[s]['B']['dice']:.2f}% Dice")
print("======================================\\n")
""")

with open("audit/ensemble/verify4/run_optimized.py", "w") as f:
    f.writelines(new_lines)
