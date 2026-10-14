"""Reproduce inductive F1: only train features fit centres/temperature; only train labels map."""
import sys,os,json,time,ast,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'audit/improve'
sys.dont_write_bytecode=True
START=time.monotonic()
from segment_ct import *
from scipy.spatial import cKDTree

def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,int):return
        p=Path(os.fsdecode(path)).resolve()
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not p.is_relative_to(HERE):raise PermissionError('Only final_pipeline writes: '+str(p))
        if os.environ.get('FITTING')=='1' and ('/labels/' in str(p) or 'scoring_masks' in str(p) or p.name.endswith('_scores.json')):raise PermissionError('Labels forbidden while fitting')
    elif event=='os.mkdir':
        p=Path(os.fsdecode(args[0])).resolve()
        if not p.exists() and not p.is_relative_to(HERE):raise PermissionError(str(p))
    elif event in ('os.remove','os.rmdir','os.rename'):
        for path in args[:2] if event=='os.rename' else args[:1]:
            if not Path(os.fsdecode(path)).resolve().is_relative_to(HERE):raise PermissionError(str(path))
    elif event=='socket.connect':raise PermissionError('Network disabled')
sys.addaudithook(guard)

def reuse(path,names):
    tree=ast.parse(path.read_text())
    for name in names:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
reuse(ROOT/'audit/scratch/forensics/common.py',['scores'])
reuse(ROOT/'audit/improve2/common.py',['surface','boundary_distances','boundary_summary'])
# Exact FORENSICS confusion expression, generalized only in dimension constants.
node=next(n for n in ast.walk(ast.parse((ROOT/'audit/scratch/forensics/task1.py').read_text())) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='conf')
expr=compile(ast.unparse(node.value).replace('minlength=16','minlength=2*K').replace('reshape(8, 2)','reshape(K, 2)'), 'FORENSICS confusion','eval')
def confusion(pr,y,K):return eval(expr,dict(np=np,pr=np.asarray(pr,dtype=np.int64),y=np.asarray(y,dtype=np.int64),K=K))
def dump(name,data):(HERE/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def records():return json.loads((OLD/'metadata.json').read_text())['records']
def identity(seed,centers,temperature,scope):
    import importlib.metadata
    weights=ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'
    return dict(schema_version=1,K=8,seed=seed,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preprocessing=contract(),temperature=temperature,backbone_weight_file=str(weights.relative_to(ROOT)),backbone_weight_sha256=file_sha256(weights),
        centers_sha256=file_sha256(centers),fitting_scope=scope,fit_labels_used=False,mapping_rule='strict majority native512 pixels; precision > 0.5; ties background',
        kmeans_parameters=dict(n_clusters=8,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd',init='k-means++',random_state=seed,copy_x=False),
        versions={s:importlib.metadata.version(s) for s in ['torch','torchvision','numpy','scipy','scikit-learn','timm','open_clip_torch','pillow','pydicom']})
def fit(seed):
    from sklearn.cluster import KMeans
    os.environ['FITTING']='1'
    rr=records();inds=[i for i,r in enumerate(rr) if r['split']=='train'];val={r['patient'] for r in rr if r['split']=='val'}
    assert len(inds)==1869 and len({rr[i]['patient'] for i in inds})==16 and not val&{rr[i]['patient'] for i in inds}
    f=np.load(OLD/'B2_448_features.npy',mmap_mode='r');path=HERE/f'inductive_seed{seed}_centers.npy'
    assert not path.exists();work=HERE/f'work_seed{seed}.npy'
    x=np.lib.format.open_memmap(work,mode='w+',dtype=np.float32,shape=(len(inds)*784,768))
    for j,i in enumerate(inds):x[j*784:(j+1)*784]=f[i]
    x.flush();print('FIT',seed,x.shape,flush=True)
    km=KMeans(n_clusters=8,random_state=seed,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd',copy_x=False).fit(x)
    gaps=[]
    for j in range(0,len(x),32768):
        d=km.transform(x[j:j+32768])**2;two=np.partition(d,1,axis=1)[:,:2];gaps.append(two[:,1]-two[:,0])
    temperature=max(1e-6,float(np.median(np.concatenate(gaps)))/float(np.log(4)))
    np.save(path,km.cluster_centers_);meta=identity(seed,path,temperature,'16 train patients; 1869 slices')
    meta.update(fit_patient_ids=sorted({rr[i]['patient'] for i in inds},key=int),ordered_fit_ids_sha256=hashlib.sha256(json.dumps([rr[i]['sample_id'] for i in inds]).encode()).hexdigest(),inertia=float(km.inertia_),iterations=int(km.n_iter_),seconds=time.monotonic()-START,producer='study.py fit')
    dump(f'inductive_seed{seed}_fit.json',meta);del x;work.unlink();print('FIT DONE',meta['seconds'],flush=True)
def mapping(seed,scope):
    rr=records();features=np.load(OLD/'B2_448_features.npy',mmap_mode='r')
    if scope=='inductive':
        meta=json.loads((HERE/f'inductive_seed{seed}_fit.json').read_text());centers=np.load(HERE/f'inductive_seed{seed}_centers.npy');conf=np.zeros((9,2),np.int64)
        y=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r')
        for i,r in enumerate(rr):
            if r['split']!='train':continue
            grids=distances(features[i],centers).argmin(1).reshape(28,28)
            conf+=confusion(inverse_grid(grids,r,outside=8),y[i],9)
        meta['mapping_labels']='16 training patients only, native512';meta['mapping_patient_ids']=meta['fit_patient_ids']
    else:
        old=json.loads((OLD/f'B2_448_s{seed}_scores.json').read_text())
        conf=sum(np.array(c,np.int64) for c in old['results']['512']['patient_confusions'].values())
        path=HERE/f'all20_seed{seed}_centers.npy';assert not path.exists()
        np.save(path,np.load(OLD/f'B2_448_s{seed}_centers.npy',allow_pickle=False))
        fit=json.loads((OLD/f'B2_448_s{seed}_fit.json').read_text());meta=identity(seed,path,fit['temperature'],'all20 patients; 2341 slices; exact saved IMPROVE centres reused')
        meta.update(mapping_labels='all20 patients; deployment artifact only, not used for inductive results',mapping_patient_ids=sorted({r['patient'] for r in rr},key=int),source_fit_file=str(OLD/f'B2_448_s{seed}_fit.json'))
    meta['liver_clusters']=np.flatnonzero(conf[:8,1]>conf[:8,0]).tolist();meta['mapping_confusion']=conf.tolist();meta['producer']='study.py mapping'
    dump(f'{scope}_seed{seed}.json',meta);print(scope,seed,'liver clusters',meta['liver_clusters'],flush=True)
def evaluate(seed):
    rr=records();meta=json.loads((HERE/f'inductive_seed{seed}.json').read_text());centers=np.load(HERE/f'inductive_seed{seed}_centers.npy')
    f=np.load(OLD/'B2_448_features.npy',mmap_mode='r');yy=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r');y224=np.load(OLD/'scoring_masks_224.npy',mmap_mode='r')
    pats=sorted({r['patient'] for r in rr if r['split']=='val'},key=int);assert len(pats)==4
    results={arm:{'patients':{},'slices':[]} for arm in ['hard','F1']}
    for pat in pats:
        inds=[i for i,r in enumerate(rr) if r['split']=='val' and r['patient']==pat]
        conf={arm:{res:np.zeros((2,2),np.int64) for res in ['224','512']} for arm in results};bd={arm:[] for arm in results};cases={arm:{'one_empty':0,'both_empty':0,'both_present':0} for arm in results}
        for i in inds:
            r=rr[i];im=load_plane(r['image']);hard,pred=predict_features(f[i],im,r,centers,meta);truth=yy[i].astype(bool);tp=np.argwhere(surface(truth))
            for arm,pr in [('hard',hard),('F1',pred)]:
                sr={}
                for res,predres,target in [('512',pr,truth),('224',resize224(pr),y224[i])]:
                    c=confusion(predres,target,2);conf[arm][res]+=c;sr[res]=scores(c,[1])
                b,case=boundary_distances(pr,truth,tp);bd[arm].append(b);cases[arm][case]+=1
                results[arm]['slices'].append(dict(index=i,stem=r['stem'],patient=pat,gt_pixels=int(truth.sum()),fp_pixels=int((pr&~truth).sum()),scores=sr))
        for arm in results:
            slices=[r for r in results[arm]['slices'] if r['patient']==pat]
            results[arm]['patients'][pat]=dict(count=len(inds),pooled={res:scores(c,[1]) for res,c in conf[arm].items()},slice_mean={res:{k:float(np.mean([r['scores'][res][k] for r in slices])) for k in ['iou','dice']} for res in ['224','512']},boundary=dict(**boundary_summary(bd[arm]),cases=cases[arm]))
        print('EVAL',seed,pat,time.monotonic()-START,flush=True)
    dump(f'inductive_seed{seed}_results.json',dict(seed=seed,producer='study.py evaluate',labels='Training labels map only; validation labels score/diagnose only',liver_clusters=meta['liver_clusters'],results=results,seconds=time.monotonic()-START))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('task',choices=['fit','mapping','evaluate']);p.add_argument('--seed',type=int,required=True);p.add_argument('--scope',default='inductive',choices=['inductive','all20']);a=p.parse_args()
    if a.task=='fit':fit(a.seed)
    elif a.task=='mapping':mapping(a.seed,a.scope)
    else:evaluate(a.seed)
