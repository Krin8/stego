import os, sys, json, torch, numpy as np, hashlib
from pathlib import Path
from scipy import ndimage as ndi
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "experiments/stego_colocated")
os.environ['STEGO_STAGE'] = 'evaluation'
import runtime as rt
from runtime import records, load_image, frozen_backbone, inverse_grid, resize224
from model import StegoHead

ROOT = Path(".").resolve()
OUT = ROOT / "audit/ensemble"
OUT.mkdir(parents=True, exist_ok=True)

def largest(mask, structure):
    lab, n = ndi.label(mask, structure=structure)
    if not n: return mask
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == counts.argmax()

def guided(image, field):
    size = 17; mi = ndi.uniform_filter(image, size=size, mode='reflect'); mp = ndi.uniform_filter(field, size=size, mode='reflect')
    var = ndi.uniform_filter(image*image, size=size, mode='reflect') - mi*mi
    cov = ndi.uniform_filter(image*field, size=size, mode='reflect') - mi*mp
    aa = cov / (var + .01); bb = mp - aa*mi
    return np.clip(ndi.uniform_filter(aa, size=size, mode='reflect')*image + ndi.uniform_filter(bb, size=size, mode='reflect'), 0, 1)

def apply_f1(im, soft):
    pred = largest(guided(im, soft) >= .5, np.ones((3,3), int))
    pred = ndi.binary_fill_holes(pred)
    yy,xx = np.mgrid[-3:4,-3:4]; disk = xx*xx+yy*yy<=9
    return ndi.binary_closing(np.pad(pred, 3), structure=disk)[3:-3, 3:-3]

from scipy.spatial.distance import cdist

def distances(f, centers):
    return cdist(f, centers)

rr = records()
train_ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
val_ids = [i for i, r in enumerate(rr) if r['split'] == 'val']

features_path = ROOT/"audit/preflight/B2_448_features.npy"
if not features_path.exists(): features_path = ROOT/"audit/improve/B2_448_features.npy"
if not features_path.exists(): features_path = rt.OLD / "B2_448_features.npy"
features = np.load(features_path)

val_y = {
    224: np.load(rt.OLD / 'scoring_masks_224.npy', mmap_mode='r'),
    512: np.load(rt.OLD / 'scoring_masks_native.npy', mmap_mode='r')
}

image_cache = {}
print("Preloading validation images...")
for i in val_ids:
    image_cache[i] = load_image(rr[i])
print("Preloaded.")

class Arm:
    def __init__(self, name, embed_fn):
        self.name = name
        self.embed_fn = embed_fn
        self.seed_results = {}
        
arms = {}

def embed_frozen(i): return np.array(features[i]).reshape(-1, 768)
arms['Frozen Baseline'] = Arm('Frozen Baseline', embed_frozen)

untrained_models = {}
for s in [0, 1, 2]:
    rt.seed_all(s)
    m = StegoHead().eval().requires_grad_(False)
    untrained_models[s] = m
def embed_untrained(s, i):
    f = torch.from_numpy(np.array(features[i])).reshape(28,28,768).permute(2,0,1)[None]
    with torch.no_grad(): z = untrained_models[s].embed(f)[0].permute(1,2,0).contiguous().numpy().reshape(-1, 70)
    return z
arms['Untrained Head'] = Arm('Untrained Head', embed_untrained)

c448_models = {}
for s in [0, 1, 2]:
    m = StegoHead().eval().requires_grad_(False)
    ckpt = torch.load(ROOT / f"experiments/stego_colocated/C448/stage1/checkpoints/seed{s}/final.pt", map_location='cpu')
    m.load_state_dict(ckpt.get('state_dict', ckpt))
    c448_models[s] = m
def embed_c448(s, i):
    f = torch.from_numpy(np.array(features[i])).reshape(28,28,768).permute(2,0,1)[None]
    with torch.no_grad(): z = c448_models[s].embed(f)[0].permute(1,2,0).contiguous().numpy().reshape(-1, 70)
    return z
arms['C448'] = Arm('C448', embed_c448)

p1_models = {}
for s in [0, 1, 2]:
    m = StegoHead().eval().requires_grad_(False)
    ckpt = torch.load(ROOT / f"experiments/stego_colocated/P1/stage1/checkpoints/seed{s}/final.pt", map_location='cpu')
    m.load_state_dict(ckpt.get('state_dict', ckpt))
    p1_models[s] = m
def embed_p1(s, i):
    f = torch.from_numpy(np.array(features[i])).reshape(28,28,768).permute(2,0,1)[None]
    with torch.no_grad(): z = p1_models[s].embed(f)[0].permute(1,2,0).contiguous().numpy().reshape(-1, 70)
    return z
arms['P1'] = Arm('P1', embed_p1)


for arm_name, arm in arms.items():
    print(f"Processing Arm: {arm_name}")
    for s in [0, 1, 2]:
        print(f"  Seed {s}...", flush=True)
        if arm_name == 'Frozen Baseline':
            centers_path = ROOT / f"audit/naming/inductive_s{s}_centers.npy"
            naming_path = ROOT / f"audit/naming/inductive_s{s}_naming.json"
        elif arm_name == 'Untrained Head':
            untrained_res = json.load(open(ROOT / "audit/colocated/verify2/untrained_results.json"))['results']
            n3_cluster = untrained_res[str(s)]['selected']['N3']
            rt.seed_all(s)
            from sklearn.cluster import KMeans
            sample = np.zeros((len(train_ids)*784, 70), dtype=np.float32)
            for j, tr_id in enumerate(train_ids):
                sample[j*784:(j+1)*784] = embed_untrained(s, tr_id)
            km = KMeans(n_clusters=8, random_state=s, n_init=3, max_iter=100, tol=1e-4, algorithm='lloyd').fit(sample)
            centers = km.cluster_centers_
            naming = {'selected': {'N3': n3_cluster}}
            d = distances(sample, centers)
            two = np.partition(d, 1, axis=1)[:, :2]
            temp = max(float(np.median(two[:,1]-two[:,0])/np.log(4)), 1e-6)
            naming['temp'] = temp
            
        elif arm_name == 'C448':
            centers_path = ROOT / f"experiments/stego_colocated/C448/stage2/predictions/seed{s}/centers.npy"
            naming_path = ROOT / f"experiments/stego_colocated/C448/stage2/predictions/seed{s}/naming.json"
        elif arm_name == 'P1':
            centers_path = ROOT / f"experiments/stego_colocated/P1/stage2/predictions/seed{s}/centers.npy"
            naming_path = ROOT / f"experiments/stego_colocated/P1/stage2/predictions/seed{s}/naming.json"
            
        if arm_name != 'Untrained Head':
            centers = np.load(centers_path)
            naming = json.load(open(naming_path))
            
            sample = np.zeros((len(train_ids)*784, centers.shape[1]), dtype=np.float32)
            for j, tr_id in enumerate(train_ids):
                sample[j*784:(j+1)*784] = arm.embed_fn(s, tr_id) if arm_name != 'Frozen Baseline' else arm.embed_fn(tr_id)
            d = distances(sample, centers)
            two = np.partition(d, 1, axis=1)[:, :2]
            temp = max(float(np.median(two[:,1]-two[:,0])/np.log(4)), 1e-6)
            naming['temp'] = temp
            
        if 'selected' in naming:
            n3 = naming['selected']['N3']
        elif 'N3' in naming and 'selected_cluster' in naming['N3']:
            n3 = naming['N3']['selected_cluster']
        else:
            n3 = naming.get('liver_cluster', None)
            
        if n3 is None:
            print(f"    Seed {s} N3 abstained. Dropping.", flush=True)
            continue
            
        arm.seed_results[s] = {'hard': {}, 'soft': {}, 'temp': naming['temp'], 'centers': centers, 'n3': n3}
        
        for i in val_ids:
            z = arm.embed_fn(i) if arm_name == 'Frozen Baseline' else arm.embed_fn(s, i)
            d = distances(z, centers)
            assignment = d.argmin(1).reshape(28, 28)
            native_hard = inverse_grid(assignment, rr[i], outside=8)
            
            weights = np.exp(-(d - d.min(1, keepdims=True)) / naming['temp'])
            weights /= weights.sum(1, keepdims=True)
            field = weights[:, n3].reshape(28, 28)
            soft_448 = inverse_grid(field, rr[i], mode='bilinear', outside=0)
            
            arm.seed_results[s]['hard'][i] = (native_hard == n3)
            arm.seed_results[s]['soft'][i] = soft_448

results = {}
for arm_name, arm in arms.items():
    print(f"Evaluating {arm_name}...", flush=True)
    results[arm_name] = {}
    valid_seeds = list(arm.seed_results.keys())
    
    for s in valid_seeds:
        results[arm_name][f"Seed {s}"] = {"224": {"hard": [], "F1": []}, "512": {"hard": [], "F1": []}}
        for i in val_ids:
            im = image_cache[i]
            hard512 = arm.seed_results[s]['hard'][i]
            f1_512 = apply_f1(im, arm.seed_results[s]['soft'][i])
            
            results[arm_name][f"Seed {s}"]["512"]["hard"].append((hard512, val_y[512][i]))
            results[arm_name][f"Seed {s}"]["512"]["F1"].append((f1_512, val_y[512][i]))
            results[arm_name][f"Seed {s}"]["224"]["hard"].append((resize224(hard512), val_y[224][i]))
            results[arm_name][f"Seed {s}"]["224"]["F1"].append((resize224(f1_512), val_y[224][i]))
            
    if len(valid_seeds) > 0:
        results[arm_name]["Soft Ensemble"] = {"224": {"hard": [], "F1": []}, "512": {"hard": [], "F1": []}}
        results[arm_name]["Hard Ensemble"] = {"224": {"hard": [], "F1": []}, "512": {"hard": [], "F1": []}}
        
        for i in val_ids:
            im = image_cache[i]
            softs = np.array([arm.seed_results[s]['soft'][i] for s in valid_seeds])
            hards = np.array([arm.seed_results[s]['hard'][i] for s in valid_seeds])
            
            mean_soft = softs.mean(axis=0)
            f1_512_soft = apply_f1(im, mean_soft)
            
            majority_hard = hards.sum(axis=0) > (len(valid_seeds) / 2.0)
            f1_512_hard = apply_f1(im, majority_hard.astype(np.float32))
            
            results[arm_name]["Soft Ensemble"]["512"]["hard"].append((mean_soft >= 0.5, val_y[512][i]))
            results[arm_name]["Soft Ensemble"]["512"]["F1"].append((f1_512_soft, val_y[512][i]))
            results[arm_name]["Soft Ensemble"]["224"]["hard"].append((resize224(mean_soft >= 0.5), val_y[224][i]))
            results[arm_name]["Soft Ensemble"]["224"]["F1"].append((resize224(f1_512_soft), val_y[224][i]))
            
            results[arm_name]["Hard Ensemble"]["512"]["hard"].append((majority_hard, val_y[512][i]))
            results[arm_name]["Hard Ensemble"]["512"]["F1"].append((f1_512_hard, val_y[512][i]))
            results[arm_name]["Hard Ensemble"]["224"]["hard"].append((resize224(majority_hard), val_y[224][i]))
            results[arm_name]["Hard Ensemble"]["224"]["F1"].append((resize224(f1_512_hard), val_y[224][i]))

out_metrics = {}
for arm_name, preds in results.items():
    out_metrics[arm_name] = {}
    for pred_name, res_dict in preds.items():
        out_metrics[arm_name][pred_name] = {}
        for res, kinds in res_dict.items():
            out_metrics[arm_name][pred_name][res] = {}
            for kind, pairs in kinds.items():
                patient_tps = {}
                patient_fps = {}
                patient_fns = {}
                slice_ious = []
                slice_dices = []
                
                for idx, (p, y) in enumerate(pairs):
                    global_i = val_ids[idx]
                    pat = rr[global_i]['patient']
                    co = np.bincount((np.asarray(p,np.int64)*2+np.asarray(y,np.int64)).ravel(), minlength=4).reshape(2,2)
                    tp = co[1,1]
                    fp = co[1,0]
                    fn = co[0,1]
                    
                    if pat not in patient_tps:
                        patient_tps[pat] = 0; patient_fps[pat] = 0; patient_fns[pat] = 0
                    patient_tps[pat] += tp; patient_fps[pat] += fp; patient_fns[pat] += fn
                    
                    slice_ious.append(tp / max(1, tp + fp + fn))
                    slice_dices.append(2*tp / max(1, 2*tp + fp + fn))
                    
                pat_ious = [t / max(1, t + f + n) for t, f, n in zip(patient_tps.values(), patient_fps.values(), patient_fns.values())]
                pat_dices = [2*t / max(1, 2*t + f + n) for t, f, n in zip(patient_tps.values(), patient_fps.values(), patient_fns.values())]
                
                out_metrics[arm_name][pred_name][res][kind] = {
                    "pooled_iou": np.mean(pat_ious), "pooled_iou_sd": np.std(pat_ious, ddof=0),
                    "pooled_dice": np.mean(pat_dices), "pooled_dice_sd": np.std(pat_dices, ddof=0),
                    "mean_slice_iou": np.mean(slice_ious), "mean_slice_iou_sd": np.std(slice_ious, ddof=0),
                    "mean_slice_dice": np.mean(slice_dices), "mean_slice_dice_sd": np.std(slice_dices, ddof=0)
                }
                
    valid_seeds = list(arms[arm_name].seed_results.keys())
    if len(valid_seeds) > 0:
        out_metrics[arm_name]["Mean over Seeds"] = {}
        for res in ["224", "512"]:
            out_metrics[arm_name]["Mean over Seeds"][res] = {}
            for kind in ["hard", "F1"]:
                metrics = {}
                for met in ["pooled_iou", "pooled_dice", "mean_slice_iou", "mean_slice_dice"]:
                    vals = [out_metrics[arm_name][f"Seed {s}"][res][kind][met] for s in valid_seeds]
                    sds = [out_metrics[arm_name][f"Seed {s}"][res][kind][met+"_sd"] for s in valid_seeds]
                    metrics[met] = np.mean(vals)
                    metrics[met+"_sd"] = np.mean(sds)
                out_metrics[arm_name]["Mean over Seeds"][res][kind] = metrics
                
temp_out = ROOT / "experiments/stego_colocated/ensemble_metrics.json"
with open(temp_out, "w") as f:
    json.dump(out_metrics, f, indent=2)

print("Ensemble computation complete.")
