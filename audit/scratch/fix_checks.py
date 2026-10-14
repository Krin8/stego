"""Bounded CPU reproduction/regression checks; no dataset/cache/checkpoint writes."""
import guard, argparse, signal, sys, ast, json
signal.alarm(110)
sys.modules['wandb'] = None
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from omegaconf import OmegaConf
torch.set_num_threads(4)
torch.manual_seed(11)
p = argparse.ArgumentParser()
p.add_argument('finding')
p.add_argument('--expect-fixed', action='store_true')
a = p.parse_args()
root = guard.ROOT
def config():
    return OmegaConf.load(root / 'src/configs/train_config.yaml')
def metrics():
    class Metric(nn.Module):
        def __init__(self, **kwargs): super().__init__()
        def add_state(self, name, default, **kwargs): setattr(self, name, default)
    return guard.definitions('src/utils.py', ['UnsupervisedMetrics'], dict(torch=torch, np=np, Metric=Metric, linear_sum_assignment=linear_sum_assignment))['UnsupervisedMetrics']
def modules(names):
    return guard.definitions('src/modules.py', names, dict(torch=torch, nn=nn, F=F, __file__=str(root / 'src/modules.py')))
def chaos_class(os_module=None):
    import os, random
    from PIL import Image
    from torch.utils.data import Dataset
    return guard.definitions('src/data.py', ['CHAOS'], dict(Dataset=Dataset, np=np, torch=torch, Image=Image, os=os_module or os, join=os.path.join, random=random))['CHAOS']
def finish(details, fixed):
    print(json.dumps(dict(finding=a.finding, fixed=bool(fixed), **details), default=str))
    if a.expect_fixed: assert fixed, details
if a.finding == 'C1':
    m = metrics()('', 2, 2, True)
    m.stats = torch.tensor([[0, 0], [0, 0], [0, 10], [90, 0]])
    score = m.compute(); lut = m.map_clusters(torch.arange(4)).tolist()
    finish(dict(lut=lut, metrics=score), lut == [0, 0, 1, 0])
elif a.finding == 'M1':
    m = metrics()('', 2, 2, True)
    m.stats = torch.tensor([[80, 0], [0, 10], [10, 0], [0, 0]])
    score = m.compute(); extra = config().extra_clusters
    finish(dict(metrics=score, extra_clusters=extra), abs(score['mIoU'] - 94.444444) < 1e-4 and extra == 0)
elif a.finding == 'C2':
    sample = modules(['sample'])['sample']
    img = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    y, x = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing='ij')
    grid = torch.stack([y, x], -1)[None].flip(2)
    error = float((sample(img, grid) - img.flip(3)).abs().max())
    weight = config().aug_alignment_weight
    finish(dict(raw_flip_error=error, weight=weight, effective_weighted_error=error * weight), weight == 0)
elif a.finding == 'C3':
    ns = modules(['norm', 'tensor_correlation', 'sample', 'ContrastiveCorrelationLoss'])
    cfg = config(); cfg.use_salience = False; cfg.feature_samples = 2; cfg.neg_samples = 4
    fn = ns['ContrastiveCorrelationLoss'](cfg)
    train = (root / 'src/train_segmentation.py').read_text()
    external = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'NegativeImageDataset' for n in ast.walk(ast.parse(train)))
    head = nn.Conv2d(3, 4, 1)
    f = torch.randn(3, 3, 2, 2); c = head(f)
    shifts = []; original_roll = torch.roll
    def traced_roll(input, shifts=1, dims=None):
        seen.append(shifts)
        return original_roll(input, shifts, dims)
    seen = []; torch.roll = traced_roll
    try:
        res = fn(f, f.flip(0), None, None, c, c.flip(0), f, c)
        loss = res[0] + res[2] + res[4].mean()
        loss.backward()
        rejected_one = False
        try: fn(f[:1], f[:1], None, None, c[:1], c[:1], f[:1], c[:1])
        except AssertionError: rejected_one = True
    finally: torch.roll = original_roll
    fixed = not external and rejected_one and all(s != 0 for s in seen) and torch.isfinite(loss)
    finish(dict(external_loader=external, shifts=seen, singleton_rejected=rejected_one, loss=float(loss.detach()), finite_gradient=bool(torch.isfinite(head.weight.grad).all())), fixed)
elif a.finding == 'M3':
    cfg = config()
    tree = ast.parse((root / 'src/train_segmentation.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LitUnsupervisedSegmenter')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    dim_if = next(n for n in init.body if isinstance(n, ast.If))
    env = dict(cfg=cfg, n_classes=2)
    exec(compile(ast.Module(body=[dim_if], type_ignores=[]), '<actual dim selection>', 'exec'), env)
    ns = modules(['BiomedCLIPFeaturizer', 'ClusterLookup'])
    bare = ns['BiomedCLIPFeaturizer'].__new__(ns['BiomedCLIPFeaturizer']); nn.Module.__init__(bare); bare.dim = env['dim']
    shape = list(bare.make_clusterer(768)(torch.randn(1, 768, 2, 2)).shape)
    torch.manual_seed(7); expected = torch.randn(2, 2)
    torch.manual_seed(7); probe = ns['ClusterLookup'](2, 2)
    gaussian = torch.equal(expected, probe.clusters)
    probe.clusters.data.copy_(torch.eye(2))
    loss, _ = probe(torch.eye(2).T.reshape(1, 2, 1, 2), None)
    controls = {k: cfg[k] for k in ['use_text_prompts', 'text_align_weight', 'rec_weight', 'crf_weight', 'correspondence_weight']}
    fixed = shape[1] == 70 and gaussian and abs(float(loss.detach()) + 1) < 1e-6 and controls == dict(use_text_prompts=False, text_align_weight=0., rec_weight=0., crf_weight=0., correspondence_weight=1.)
    finish(dict(head_shape=shape, gaussian_init=gaussian, probe_loss=float(loss.detach()), controls=controls), fixed)
elif a.finding == 'M9':
    sal = torch.zeros(1, 4, 6); sal[0, 3, 5] = 1
    coord = modules(['sample_nonzero_locations'])['sample_nonzero_locations'](sal, [1, 1, 1, 2]).flatten().tolist()
    cfg = config()
    finish(dict(raw_bottom_right=coord, use_salience=cfg.use_salience, feature_samples=cfg.feature_samples), not cfg.use_salience and cfg.feature_samples == 11)
elif a.finding == 'M4':
    import os, random
    from types import SimpleNamespace
    from torch.utils.data import Dataset
    class FakeData:
        samples = [('image', 'mask', 'CT', '1')] * 8
        def __len__(self): return 8
        def __getitem__(self, i): return (torch.full((3, 2, 2), float(i)), torch.zeros(2, 2), torch.ones(2, 2))
    class Cache(dict):
        def __enter__(self): return self
        def __exit__(self, *args): pass
    cache = Cache(nns=np.tile(np.arange(8), (8, 1)))
    fake_np = SimpleNamespace(load=lambda *args, **kw: cache, random=np.random)
    fake_os = SimpleNamespace(path=SimpleNamespace(join=os.path.join, exists=lambda p: True))
    rng = SimpleNamespace(seed=lambda s: None, randint=lambda lo, hi: hi, choice=random.choice)
    ns = guard.definitions('src/data.py', ['ContrastiveSegDataset'], dict(Dataset=Dataset, CHAOS=lambda **kw: FakeData(), torch=torch, np=fake_np, os=fake_os, random=rng))
    cls = ns['ContrastiveSegDataset']; rejected = False
    try: cls('virtual', 'chaos', 'five', 'train', None, None, config(), num_neighbors=7)
    except ValueError as e: rejected = 'metadata' in str(e).lower()
    d = cls.__new__(cls); d.dataset=FakeData(); d.nns=torch.arange(8)[None]; d.num_neighbors=7; d.pos_images=True
    d.extra_transform=None; d.aug_geometric_transform=None; d.aug_photometric_transform=None; d.mask=False
    rank = int(d[0]['img_pos'][0, 0, 0])
    d.nns = torch.tensor([[0, 99]]); invalid = False
    try: d[0]
    except AssertionError: invalid = True
    d.nns = torch.tensor([[0]]); empty = False
    try: d[0]
    except AssertionError: empty = True
    tree = ast.parse((root/'src/precompute_knns.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'no_ap_model' and n.value.func.attr == 'eval']
    model = nn.Dropout2d(.5); env = dict(no_ap_model=model)
    for node in calls: exec(compile(ast.Module(body=[node], type_ignores=[]), '<precompute mode>', 'exec'), env)
    fixed = rejected and rank == 7 and invalid and empty and not model.training
    finish(dict(missing_metadata_rejected=rejected, highest_neighbor_rank=rank, invalid_index_rejected=invalid, no_neighbor_rejected=empty, precompute_eval=not model.training), fixed)
elif a.finding == 'M7':
    from types import SimpleNamespace
    probe = modules(['ClusterLookup'])['ClusterLookup'](2, 2)
    probe.clusters.data.copy_(torch.eye(2))
    code = torch.tensor([[[[1., 1.]], [[0., .8]]]])
    tree = ast.parse((root/'src/eval_segmentation.py').read_text())
    statements = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'cluster_probs' for t in n.targets for x in ast.walk(t)):
            statements.append(n)
    env = dict(model=SimpleNamespace(cluster_probe=probe), code=code, torch=torch)
    exec(compile(ast.Module(body=sorted(statements, key=lambda n:n.lineno), type_ignores=[]), '<eval unary>', 'exec'), env)
    expected = F.log_softmax(F.normalize(code, dim=1) * 2, dim=1)
    error = float((env['cluster_probs']-expected).abs().max().detach())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == 'cfg.run_crf')
    calls = []
    class Result:
        def argmax(self, dim): return self
        def cpu(self): calls.append('cpu'); return self
        def to(self, device): calls.append(str(device)); return self
    env.update(cfg=SimpleNamespace(run_crf=True), batched_crf=lambda *args: Result(), pool=None, img=None, linear_probs=None, label=SimpleNamespace(device='label-device'))
    exec(compile(ast.Module(body=[branch], type_ignores=[]), '<CRF device routing>', 'exec'), env)
    finish(dict(unary_max_error=error, crf_destination_calls=calls, cuda_execution='UNVERIFIED; CPU routing spy only'), error < 1e-6 and calls == ['label-device', 'label-device'])
elif a.finding == 'M8':
    tree = ast.parse((root/'src/train_segmentation.py').read_text())
    trainer = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Trainer')
    value = next(k.value for k in trainer.keywords if k.arg == 'max_steps')
    cfg = config(); cfg.max_steps = 7
    limit = eval(compile(ast.Expression(value), '<Trainer max_steps>', 'eval'), dict(cfg=cfg))
    step = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
    optimizer_steps = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'step' and isinstance(n.func.value, ast.Name) and n.func.value.id.endswith('_optim') for n in ast.walk(step))
    completed_batches = (limit + optimizer_steps - 1) // optimizer_steps
    finish(dict(requested_head_updates=7, trainer_optimizer_step_limit=limit, optimizer_steps_per_batch=optimizer_steps, actual_head_updates=completed_batches), limit == 21 and optimizer_steps == 3 and completed_batches == 7)
elif a.finding == 'M10':
    from types import SimpleNamespace
    tree = ast.parse((root/'src/train_segmentation.py').read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'linear_loss' for t in n.targets))
    logits = torch.randn(4, 2, requires_grad=True)
    env = dict(self=SimpleNamespace(linear_probe_loss_fn=nn.CrossEntropyLoss()), linear_logits=logits, flat_label=torch.full((4,), -1, dtype=torch.long), mask=torch.zeros(4, dtype=torch.bool))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<empty label CE>', 'exec'), env)
    loss = env['linear_loss']; loss.backward()
    fn = guard.definitions('src/demo_segmentation.py', ['apply_cluster_mapping'], dict(np=np))['apply_cluster_mapping']
    rejected = False
    try: fn(None, np.ones((2, 2), dtype=int), None, 2)
    except RuntimeError: rejected = True
    mapped = fn(None, np.ones((2, 2), dtype=int), np.array([0, 1]), 2)
    fixed = torch.isfinite(loss) and loss.item() == 0 and rejected and np.all(mapped == 1)
    finish(dict(empty_loss=float(loss.detach()), zero_gradient=bool((logits.grad == 0).all()), missing_mapping_rejected=rejected, valid_mapping_unchanged=bool(np.all(mapped == 1))), fixed)
elif a.finding == 'M5':
    import os
    from types import SimpleNamespace
    fake_os = SimpleNamespace(path=SimpleNamespace(dirname=os.path.dirname, abspath=os.path.abspath, isdir=lambda p: True, exists=lambda p: True), listdir=lambda p: ['CT_28_0110.png', 'CT_p28_s2037.png'])
    cls = chaos_class(fake_os); d=cls.__new__(cls)
    d.root='virtual'; d.image_set='train'; d.modality='CT'; d.samples=[]
    mixed=False
    try: d._collect_preprocessed_samples()
    except ValueError as e: mixed='Mixed preprocessing' in str(e)
    raw=cls.__new__(cls); raw.use_preprocessed_data=False; raw.samples=[('virtual.dcm', None, 'T1DUAL', '1')]
    called=[]; raw._load_dicom_as_pil=lambda p: called.append(p)
    raw.transform=lambda x: torch.ones(3, 2, 2)
    blocked=False
    try: raw[0]
    except NotImplementedError: blocked=True
    finish(dict(mixed_families_rejected=mixed, raw_mri_blocked=blocked, ct_window_called=bool(called)), mixed and blocked and not called)
elif a.finding == 'M6':
    cls = chaos_class(); d=cls.__new__(cls)
    d.use_preprocessed_data=False; d.samples=[('virtual.nii.gz', None, 'CT', '1')]
    d.transform=None; d.target_transform=None; called=[]
    def dicom(*args):
        called.append(args[0]); return torch.ones(3, 2, 2), torch.zeros(2, 2)
    d._preprocess_ct_slice=dicom
    rejected=False
    try: d[0]
    except NotImplementedError as e: rejected='NIfTI' in str(e)
    finish(dict(nifti_rejected_explicitly=rejected, reached_dicom=bool(called)), rejected and not called)
elif a.finding == 'm2':
    from PIL import Image
    from torchvision import transforms as T
    from types import SimpleNamespace
    sys.modules['utils'] = SimpleNamespace(normalize=None)  # unused import in isolated transform
    ns=guard.definitions('src/vis_corr.py', ['get_transform'], dict(torch=torch, np=np, T=T, Image=Image))
    normalize = ns['get_transform'](224, False, 'center').transforms[-1]
    expected_mean=[.48145466,.4578275,.40821073]; expected_std=[.26862954,.26130258,.27577711]
    titles=(root/'src/vis_corr.py').read_text(); pixel=(root/'src/test_timm.py').read_text()
    fixed=list(normalize.mean)==expected_mean and list(normalize.std)==expected_std and 'Random other-patient slice' in titles and 'RGB pixel cosine' in pixel
    finish(dict(mean=normalize.mean, std=normalize.std, sampled_pair_label='Random other-patient slice' in titles, rgb_label='RGB pixel cosine' in pixel), fixed)
elif a.finding == 'm3':
    from torchvision import transforms as T
    ns = guard.definitions('src/utils.py', ['UnNormalize', 'unnorm', 'prep_for_plot'], dict(torch=torch, F=F, T=T))
    mean=torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
    raw=torch.arange(12, dtype=torch.float32).reshape(3,2,2)/12
    out=ns['prep_for_plot']((raw-mean)/std, rescale=False)
    error=float((out-raw.permute(1,2,0)).abs().max())
    # Force exactly constant unnormalizer output to exercise zero-range plotting.
    ns['unnorm']=lambda t: torch.ones_like(t)
    flat=ns['prep_for_plot'](torch.zeros(3,2,2))
    finish(dict(channel_max_error=error, constant_finite=bool(torch.isfinite(flat).all())), error < 1e-6 and torch.isfinite(flat).all())
elif a.finding == 'm4':
    cfg=config()
    unsupported=['use_nns','use_fit_model','backbone_lr','hungarian_weight','reset_probe_steps','checkpoint_freq','repulsion_weight','stop_at_zero']
    remaining=[k for k in unsupported if k in cfg]
    tree=ast.parse((root/'src/test_biomedclip.py').read_text())
    bad_import=any(isinstance(n, ast.ImportFrom) and n.module=='configs.train_config' for n in ast.walk(tree))
    main=next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    assignment=next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='cfg' for t in n.targets))
    actual=None
    if not bad_import:
        from pathlib import Path
        env=dict(OmegaConf=OmegaConf, Path=Path, __file__=str(root/'src/test_biomedclip.py'))
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<smoke config>', 'exec'),env)
        actual=env['cfg'].dim
    finish(dict(unsupported_defaults=remaining, nonexistent_import=bad_import, smoke_config_dim=actual), not remaining and not bad_import and actual==70)
elif a.finding == 'm5':
    from PIL import Image
    from torchvision import transforms as T
    from types import SimpleNamespace
    ns=guard.definitions('src/utils.py', ['normalize','ToTargetTensor','identity_transform','get_transform'], dict(torch=torch,np=np,T=T,Image=Image))
    sys.modules['utils']=SimpleNamespace(normalize=ns['normalize'])
    diag=guard.definitions('src/vis_corr.py',['get_transform'],dict(torch=torch,np=np,T=T,Image=Image))
    modes={name:str(f(224,label,'center').transforms[0].interpolation) for name,f,label in [('image',ns['get_transform'],False),('mask',ns['get_transform'],True),('diagnostic',diag['get_transform'],False)]}
    fixed=all(f(224,label,'center').transforms[0].interpolation==(T.InterpolationMode.NEAREST if label else T.InterpolationMode.BILINEAR) for f,label in [(ns['get_transform'],False),(ns['get_transform'],True),(diag['get_transform'],False)])
    finish(dict(interpolation=modes),fixed)
else:
    raise ValueError(a.finding)
