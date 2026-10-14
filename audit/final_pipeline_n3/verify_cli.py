"""Exercise real CLI/check mode and compare DICOM/PNG outputs without modifying inputs."""
from study import *
import shutil
check=json.loads((HERE/'check_reference.json').read_text());pv=json.loads((HERE/'package_verification.json').read_text())
test=HERE/'cli_test_verified';test.mkdir();png=test/'png_input';dicom=test/'dicom_input';png.mkdir();dicom.mkdir()
for row in check['samples'][:2]:shutil.copyfile(ROOT/row['image'],png/(row['stem']+'.png'))
for row,d in zip(check['samples'][:2],pv['dicom_checks'][:2]):shutil.copyfile(d['path'],dicom/(row['stem']+'.dcm'))
results=[]
def call(args,name,success=True):
    cmd=[sys.executable,'-B',str(HERE/'segment_ct.py'),*args]
    with (HERE/'commands.txt').open('a') as f:f.write(' '.join(cmd)+'\n')
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    (test/f'{name}.stdout').write_text(r.stdout);(test/f'{name}.stderr').write_text(r.stderr)
    assert (r.returncode==0)==success,(name,r.stderr)
    results.append(dict(name=name,returncode=r.returncode,expected_success=success))
for seed in [0,1,2]:call(['--check','--seed',str(seed)],f'check_seed{seed}')
for name,path in [('png',png),('dicom',dicom)]:call(['--input',str(path),'--output',str(test/f'{name}_output')],name)
for row in check['samples'][:2]:
    for kind in ['mask','overlay']:
        file=row['stem']+'_'+kind+'.png'
        assert (test/'png_output'/file).read_bytes()==(test/'dicom_output'/file).read_bytes()
call(['--input',str(png),'--output',str(test/'png_output')],'refuse_overwrite',False)
# F2 test: three adjacent CT slices in patient5 with physical order provided explicitly.
rr=records();r5=sorted([r for r in rr if r['patient']=='5'],key=lambda r:r['physical_position'])[40:43]
f2input=test/'f2_input';f2input.mkdir();order=[]
for r in r5:
    shutil.copyfile(r['image'],f2input/(r['stem']+'.png'));order.append(dict(file=r['stem']+'.png',series_id='CT5',physical_position=r['physical_position']))
orderpath=test/'order.json';orderpath.write_text(json.dumps(order))
call(['--input',str(f2input),'--output',str(test/'f2_output'),'--f2','--order-manifest',str(orderpath)],'f2')
call(['--input',str(f2input),'--output',str(test/'f2_invalid_output'),'--f2'],'f2_refuse_rank_order',False)
# Unsupported PNG contract rejection.
bad=test/'bad_png';bad.mkdir();Image.fromarray(np.zeros((512,512),np.uint8)).save(bad/'eightbit.png')
call(['--input',str(bad),'--output',str(test/'bad_png_output')],'refuse_eightbit',False)
dump('cli_verification.json',dict(producer='verify_cli.py',tests=results,dicom_png_mask_overlay_bitwise_identical=True,check_seeds=[0,1,2],native_output_shape=[512,512]))
print('CLI PASS',flush=True)
