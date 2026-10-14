from study import *
# Frozen body/filter functions are copied exactly, rather than approximately reproduced.
prov=json.loads((HERE/'function_provenance.json').read_text());tree=ast.parse((HERE/'segment_ct.py').read_text())
for name,p in prov.items():
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    s=ast.get_source_segment((HERE/'segment_ct.py').read_text(),node)
    assert hashlib.sha256(s.encode()).hexdigest()==p['sha256'],name
# Optional median flag must reproduce the physical-order cached-feature result.
rr=records();chosen=sorted([r for r in rr if r['patient']=='5'],key=lambda r:r['physical_position'])[40:43]
features=np.load(OLD/'B2_448_features.npy',mmap_mode='r');by_id={r['sample_id']:i for i,r in enumerate(rr)}
centers,meta=load_artifact(0);f1=[]
for r in chosen:
    _,pred=predict_features(features[by_id[r['sample_id']]],load_plane(r['image']),r,centers,meta);f1.append(pred)
expected=ndi.median_filter(np.stack(f1),size=(3,1,1),mode='nearest')
for r,pred in zip(chosen,expected):
    with Image.open(HERE/'cli_test_verified/f2_output'/f"{r['stem']}_mask.png") as image:actual=np.asarray(image)>0
    assert np.array_equal(pred,actual)
cli=json.loads((HERE/'cli_verification.json').read_text());assert all((x['returncode']==0)==x['expected_success'] for x in cli['tests'])
for scope in ['inductive','all20']:
    for seed in [0,1,2]:
        path=HERE/f'{scope}_seed{seed}_centers.npy';m=json.loads((HERE/f'{scope}_seed{seed}.json').read_text())
        assert np.load(path,allow_pickle=False).shape==(8,768)
        assert file_sha256(path)==m['centers_sha256']
        assert m['preprocessing']==contract()
run_state=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
assert run_state['git_diff']==''
# Record every project's source file digest for future drift checks.
run_state['source_file_sha256']={str(p.relative_to(ROOT)):file_sha256(p) for p in sorted((ROOT/'src').glob('*.py'))}
dump('final_verification.json',dict(producer='final_verify.py',function_provenance_exact=True,f2_physical_order_prediction_exact=True,all_artifact_digests_verified=True,all_check_seeds_pass=True,source_state=run_state))
with (HERE/'FINAL_PIPELINE.md').open('a') as f:
 f.write('\n## Package verification\n\n`package_check.py`: five live 448 feature arrays exactly match IMPROVE (maximum absolute difference0); five corresponding raw DICOM/uint16 PNG intensity planes match exactly. `verify_cli.py`: `--check` passes for seeds0/1/2; real PNG and DICOM folder inference produces byte-identical masks and overlays; overwrite, ambiguous8-bit PNG and physically-unordered F2 inputs are rejected. `final_verify.py`: optional F2 predictions match the physical-order median exactly; all copied F1 functions and artifact digests match. No tracked project-source diff. Evidence: `package_verification.json`, `cli_verification.json`, `final_verification.json`.\n')
print('FINAL PASS')
