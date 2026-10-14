"""Parse train-only monitor logs into a compact collapse trajectory record."""
from runtime import *
import re

def rows(path):
    text = path.read_text()
    out=[]
    for line in text.splitlines():
        chunks = [line]
        if r"\n" in line:
            chunks = line.split(r"\n")
        for chunk in chunks:
            m = re.search(r"\{.*\}", chunk)
            if not m: continue
            try: value=json.loads(m.group(0))
            except json.JSONDecodeError: continue
            if "monitor_loss" in value: out.append(value)
    # A resumed seed-1 log contains an interrupted duplicate segment. Keep the
    # last record for each batch, which is the completed run's record.
    by_batch={int(r["batch"]):r for r in out}
    return [by_batch[k] for k in sorted(by_batch)]

summary={}
for seed in [0,1,2]:
    complete=read(HERE/f"logs/seed{seed}_complete.json")
    history=rows(HERE/f"logs/seed{seed}_train.jsonl")
    bad=[r for r in history if r.get("collapse_detected")]
    summary[str(seed)]=dict(batches=complete["batches"],labels_opened=complete["labels_opened"],validation_images_used=complete["validation_images_used"],monitor_count=len(history),first_bad_batch=bad[0]["batch"] if bad else None,final=history[-1],trajectory=[dict(batch=r["batch"],code_std=r["code_std"],cross_image_cosine=r["cross_image_cosine"],consecutive_bad_checks=r["consecutive_bad_checks"],collapse_detected=r["collapse_detected"],monitor_loss=r["monitor_loss"]) for r in history])
assert all(v["batches"]==5000 and not v["labels_opened"] and v["validation_images_used"]==0 for v in summary.values())
dump("logs/collapse_trajectories.json",dict(config_sha256=sha(HERE/"configs/locked.json"),labels_opened=False,seeds=summary))
print(json.dumps({k:{"first_bad_batch":v["first_bad_batch"],"final":v["final"]} for k,v in summary.items()},indent=2))
