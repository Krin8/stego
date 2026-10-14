"""Resume interrupted seed 1 under the locked protocol."""
import os
os.environ["STEGO_STAGE"] = "training"
from runtime import *
from model import StegoHead, objective
from reference_loss import ContrastiveCorrelationLoss
c = read(HERE/"configs/locked.json")
assert c["status"] == "LOCKED"
assert sha(HERE/"configs/locked.json") == (HERE/"configs/locked.sha256").read_text().strip()
check = read(HERE/"logs/preflight.json")
assert check["passed"] and check["backbone_trainable"] == 0
rr = records()
ids = [i for i,r in enumerate(rr) if r["split"] == "train"]
assert len(ids) == 1869
features = np.load(OLD/"B2_448_features.npy", mmap_mode="r")
neighbors = np.load(HERE/"logs/train_neighbors.npy", allow_pickle=False)
state = torch.load(HERE/"checkpoints/seed1/last.pt", map_location="cpu", weights_only=True)
start = int(state["step"])
assert 0 < start < c["batches"] and state["config_sha256"] == sha(HERE/"configs/locked.json")
model = StegoHead()
model.load_state_dict(state["state_dict"], strict=True)
criterion = ContrastiveCorrelationLoss(SimpleNamespace(**c))
opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
rng = np.random.default_rng(1)
monitor_rng = np.random.default_rng(c["monitor_seed"])
patients = sorted({rr[i]["patient"] for i in ids}, key=int)
monitor = [int(monitor_rng.choice([j for j,i in enumerate(ids) if rr[i]["patient"] == p])) for p in patients]
monitor_pos = [int(neighbors[j, monitor_rng.integers(7)]) for j in monitor]
def batch(rows):
    return torch.from_numpy(np.stack([features[ids[int(j)]] for j in rows])).reshape(len(rows),28,28,768).permute(0,3,1,2).contiguous()
fixed = batch(monitor), batch(monitor_pos)
best = float(state.get("criterion", float("inf"))); best_step = start; bad = 0
order = rng.permutation(len(ids)); cursor = 0
log = HERE/"logs/seed1_train.jsonl"
for step in range(start+1, c["batches"]+1):
    if cursor+c["batch_size"] > len(order):
        order = rng.permutation(len(ids)); cursor = 0
    selected = order[cursor:cursor+c["batch_size"]]; cursor += c["batch_size"]
    positive = [neighbors[j, rng.integers(c["num_neighbors"])] for j in selected]
    model.train()
    f,code = model(batch(selected)); fp,cp = model(batch(positive))
    loss,parts = objective(criterion, SimpleNamespace(**c), f, fp, code, cp)
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    row = dict(batch=step, loss=float(loss.detach()), **{k:float(v.detach()) for k,v in parts.items()})
    if step%c["monitor_every"] == 0 or step == c["batches"]:
        model.eval()
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(c["monitor_seed"])
            fm,zm = model(fixed[0]); fp,zp = model(fixed[1])
            value,_ = objective(criterion, SimpleNamespace(**c), fm, fp, zm, zp)
            z = F.normalize(zm, dim=1); std = float(z.flatten(2).std(2).mean())
            means = z.flatten(2).mean(2)
            cos = float(((means@means.T)[~torch.eye(len(means),dtype=torch.bool)]).mean())
        bad = bad+1 if cos>.99 or std<.01 else 0
        row.update(monitor_loss=float(value),code_std=std,cross_image_cosine=cos,consecutive_bad_checks=bad,collapse_detected=bad>=3)
        saved = dict(state_dict=model.state_dict(),seed=1,step=step,criterion=float(value),config_sha256=sha(HERE/"configs/locked.json"),source_sha256=check["source_sha256"],backbone_sha256=check["backbone_sha256"],train_patients=patients,monitor_indices=[ids[j] for j in monitor],resumed_from_batch=start)
        if float(value) < best:
            best=float(value); best_step=step; torch.save(saved,HERE/"checkpoints/seed1/best.pt")
        torch.save(saved,HERE/"checkpoints/seed1/last.pt"); print(json.dumps(row), flush=True)
    with log.open("a") as stream: stream.write(json.dumps(row)+chr(10))
dump("logs/seed1_resume.json",dict(start_batch=start,target_batch=c["batches"],config_sha256=sha(HERE/"configs/locked.json"),optimizer_state_restored=False,rng_state_restored=False,labels_opened=False))
dump("logs/seed1_complete.json",dict(seed=1,batches=c["batches"],best_step=best_step,best_loss=best,labels_opened=False,validation_images_used=0,train_patients=patients,resumed_from_batch=start,optimizer_state_restored=False,rng_state_restored=False))
ledger("logs/seed1_training_access.json")
