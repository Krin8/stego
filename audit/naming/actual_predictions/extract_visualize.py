"""Visualize actual sealed/Stage-B arrays; never read or reconstruct from a PDF."""
import os,sys
from pathlib import Path
os.environ['NAMING_STAGE']='B';sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1];DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from common import *
# Explicit prohibition on opening PDFs, including the old contour figure.
def no_pdf(event,args):
 if event=='open' and not isinstance(args[0],int):
  p=Path(os.fsdecode(args[0]));mode=args[1]
  if p.suffix.lower()=='.pdf':raise PermissionError('No PDF inputs or reconstruction allowed')
sys.addaudithook(no_pdf)
manifest=sealed_manifest();assert manifest['selections']['inductive_s0']==dict(N1=2,N2=2,N3=2)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
records=json.loads((BASE/'image_records.json').read_text());names=json.loads((BASE/'inductive_s0_naming.json').read_text());fixed=json.loads((OLD/'figure_selection.json').read_text())['eight_indices']
grids=np.load(BASE/'inductive_s0_grids.npy',mmap_mode='r');panels=np.load(BASE/'inductive_s0_figure_masks.npz',allow_pickle=False);repeat=np.load(BASE/'publication/reproduction/inductive_s0_figure_masks.npz',allow_pickle=False)
results=json.loads((BASE/'inductive_s0_scores.json').read_text());res_repeat=json.loads((BASE/'publication/reproduction/inductive_s0_scores.json').read_text());y512=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r');y224=np.load(OLD/'scoring_masks_224.npy',mmap_mode='r');features=np.load(OLD/'B2_448_features.npy',mmap_mode='r');centers=np.load(BASE/'inductive_s0_centers.npy',allow_pickle=False)
colors=['#4477aa','#ee6677','#228833','#ccbb44','#66ccee','#aa3377','#bbbbbb','#663399','#111111'];cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,9.5,1),9)
ct=[];gt=[];cluster_native=[];hard=[];f1=[];rows=[];outputs=[]
def save_npy(name,x):
 path=DEST/name
 with path.open('xb') as f:np.save(f,x,allow_pickle=False)
 outputs.append(path);return str(path)
def save_png(path,x):
 with path.open('xb') as f:Image.fromarray(x).save(f,format='PNG')
 outputs.append(path)
for i in fixed:
 r=records[i];stem=r['stem'];im=raw_image(r);shape=tuple(r['shape']);assert shape==(512,512) and im.shape==shape
 raw=grids[i].copy();assert raw.shape==(28,28) and raw.min()>=0 and raw.max()<8
 # Prove saved cluster assignments originate from the fixed feature/centre path.
 recomputed=distances(features[i],centers).argmin(1).reshape(28,28).astype(np.uint8);assert np.array_equal(raw,recomputed)
 native=inverse_grid(raw,r,mode='nearest',outside=8).astype(np.uint8)
 h=panels[f'N1_hard_{i}'].copy();f=panels[f'N1_F1_{i}'].copy();assert np.array_equal(h,native==2)
 for rule in ['N1','N2','N3']:
  for kind,a in [('hard',h),('F1',f)]:
   assert np.array_equal(a,panels[f'{rule}_{kind}_{i}']);assert np.array_equal(a,repeat[f'{rule}_{kind}_{i}'])
 # Additional direct reproduction, no backbone extraction/refit/training.
 hh,ff=predict_features(features[i],im,r,centers,dict(liver_clusters=[2],temperature=names['temperature']));assert np.array_equal(h,hh) and np.array_equal(f,ff)
 for a in [h,f,y512[i]]:assert a.shape==shape and set(np.unique(a)).issubset({0,1})
 metrics={}
 for size,truth,pr in [('512',y512[i],f),('224',y224[i],resize224(f))]:
  actual=scores(confusion(pr,truth,2),[1]);expected=next(s for s in results['rules']['N1']['results']['F1'][size]['patients'][r['patient']]['slice_scores'] if s['index']==i)
  expected2=next(s for s in res_repeat['rules']['N1']['results']['F1'][size]['patients'][r['patient']]['slice_scores'] if s['index']==i)
  for k in ['iou','dice','tp','fp','fn']:assert actual[k]==expected[k]==expected2[k],(stem,size,k)
  metrics[size]=dict(actual=actual,expected={k:expected[k] for k in ['iou','dice','tp','fp','fn']},absolute_iou_error=0.,absolute_dice_error=0.)
 folder=DEST/stem;folder.mkdir(exist_ok=True)
 paths={}
 for name,a in [('clusters_native_ids',native),('selected_liver_hard',h*255),('final_F1',f*255),('ground_truth',y512[i]*255)]:
  path=folder/(name+'.png');save_png(path,a.astype(np.uint8));paths[name]=str(path)
 # Color image purely for display. Raw ID PNG retains IDs0..8, not RGB values.
 rgb=(np.asarray(cmap(norm(native)))[...,:3]*255).round().astype(np.uint8);save_png(folder/'clusters_native_colors.png',rgb)
 overlay=np.repeat(im[...,None],3,axis=2);use=f.astype(bool);overlay[use]=.6*overlay[use]+.4*np.array([.1,.9,.35]);save_png(folder/'final_F1_overlay.png',np.rint(255*np.clip(overlay,0,1)).astype(np.uint8))
 save_png(folder/'original_CT_windowed.png',np.rint(im*65535).astype(np.uint16))
 ct.append(im);gt.append(y512[i].copy());cluster_native.append(native);hard.append(h);f1.append(f);rows.append(dict(index=i,stem=stem,patient=r['patient'],shape=list(shape),crop_box=r['box'],selected_cluster=2,raw_ids_present=np.unique(raw).tolist(),outside_crop_pixels=int((native==8).sum()),metrics=metrics,files=paths))
ct=np.stack(ct);gt=np.stack(gt);cluster_native=np.stack(cluster_native);hard=np.stack(hard);f1=np.stack(f1)
paths={name:save_npy(name+'.npy',value) for name,value in [('cluster_grid_28',grids[fixed].copy()),('cluster_assignments_native_512',cluster_native),('selected_liver_hard_512',hard),('final_F1_512',f1),('ground_truth_512',gt)]}
# Fixed native coordinate axes: every image is512×512 with no flip/transpose/crop in the display.
for cols,name in [(5,'actual_predictions_five_columns'),(3,'actual_predictions_F1_overlay')]:
 fig,axes=plt.subplots(8,cols,figsize=(4*cols,3.6*8));fig.subplots_adjust(left=.035,right=.99,bottom=.04,top=.955,hspace=.19,wspace=.025)
 for row,r in enumerate(rows):
  axes[row,0].imshow(ct[row],cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[row,0].set_title(r['stem'],fontsize=11)
  axes[row,1].imshow(gt[row],cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[row,1].set_title('Ground truth — filled binary',fontsize=11)
  if cols==5:
   axes[row,2].imshow(cluster_native[row],cmap=cmap,norm=norm,interpolation='nearest');axes[row,2].set_title('Raw K=8 clusters',fontsize=11)
   axes[row,3].imshow(hard[row],cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[row,3].set_title('Selected C2 — hard binary',fontsize=11)
   axes[row,4].imshow(f1[row],cmap='gray',vmin=0,vmax=1,interpolation='nearest');a=axes[row,4]
  else:
   axes[row,2].imshow(ct[row],cmap='gray',vmin=0,vmax=1,interpolation='nearest');rgba=np.zeros((512,512,4));rgba[f1[row].astype(bool)]=[.1,.9,.35,.4];axes[row,2].imshow(rgba,interpolation='nearest');a=axes[row,2]
  q=r['metrics']['512']['actual'];a.set_title(f'Final F1 — {"filled binary" if cols==5 else "filled overlay"}\nIoU {100*q["iou"]:.3f}% | Dice {100*q["dice"]:.3f}%',fontsize=10)
  for a in axes[row]:a.set_xlim(-.5,511.5);a.set_ylim(511.5,-.5);a.set_aspect('equal');a.axis('off')
 fig.suptitle('Actual saved predictions · inductive seed 0 · N1 = N2 = N3 · liver = C2\nNative 512 × 512 geometry; CT window [−150, 250] HU',fontsize=17,y=.987)
 if cols==5:
  handles=[Patch(facecolor=colors[c],label=f'C{c}'+(' (liver)' if c==2 else '')) for c in range(8)]+[Patch(facecolor=colors[8],label='Outside crop (ID8 sentinel; not a cluster)')];fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=11,bbox_to_anchor=(.5,.007))
 path=DEST/(name+'.png');assert not path.exists();fig.savefig(path,dpi=160,facecolor='white');plt.close(fig);outputs.append(path)
# Verify serialized native arrays and per-slice PNGs are the exact arrays plotted.
assert np.array_equal(np.load(paths['final_F1_512']),f1)
for j,r in enumerate(rows):
 with Image.open(r['files']['final_F1']) as x:assert np.array_equal(np.array(x)>0,f1[j].astype(bool))
 with Image.open(r['files']['selected_liver_hard']) as x:assert np.array_equal(np.array(x)>0,hard[j].astype(bool))
 with Image.open(r['files']['clusters_native_ids']) as x:assert np.array_equal(np.array(x),cluster_native[j])
sealed_manifest()
source_paths=[BASE/'inductive_s0_grids.npy',BASE/'inductive_s0_figure_masks.npz',BASE/'inductive_s0_scores.json',BASE/'publication/reproduction/inductive_s0_figure_masks.npz',BASE/'publication/reproduction/inductive_s0_scores.json',BASE/'inductive_s0_centers.npy',BASE/'image_records.json']
record=dict(producer=str(Path(__file__).resolve()),protocol='inductive seed0 N1=N2=N3; unchanged hard/F1',selected_cluster=2,manifest_sha256=sha(BASE/'stageA_manifest.json'),sealed_hashes_unchanged=True,pdf_files_read=0,no_refitting_or_backbone_forward=True,slice_order=fixed,arrays=paths,source_sha256={str(p):sha(p) for p in source_paths},output_sha256={str(p):sha(p) for p in outputs},geometry='Native512×512, original orientation; saved28×28 crop assignments inverse-mapped by original nearest inverse_grid; outside crop sentinel8 is not a ninth cluster. Hard/native=(IDs==2). Binary NPY values0/1; binary PNG values0/255.',labels='Ground truth display and IoU/Dice verification only; no selection or parameter adjustment',slices=rows,all_masks_exactly_match_original_and_repeated_StageB=True,all_metrics_exact_at_224_and_512=True)
(DEST/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
lines=['# Actual prediction arrays and visualizations','', 'Inductive seed0, N1=N2=N3, selected liver cluster **C2**. No PDF was read. No masks were reconstructed. All42 StageA hashes remain unchanged.','', 'Source cluster grids: '+str(BASE/'inductive_s0_grids.npy')+'; shape2341×28×28. Saved masks: '+str(BASE/'inductive_s0_figure_masks.npz')+'; keys N1_hard_<index> and N1_F1_<index> (N2/N3 identical). Each selected grid also matches argmin squared-distance to the existing centres; each saved mask matches an unchanged predict_features call from existing frozen features.','', 'Native geometry uses the original inverse_grid crop metadata with nearest interpolation for hard cluster IDs. Outside-crop ID8 is a sentinel, not another cluster. All masks are filled native512×512 arrays. Overlay alpha.4 is for display only. NPY arrays are in the exact eight-slice order listed below; per-slice PNGs have no resizing.','', '| Slice | Global index | F1 native IoU % | F1 native Dice % | Absolute error vs StageB |','|---|---:|---:|---:|---:|']
for r in rows:
 q=r['metrics']['512']['actual'];lines.append(f"| {r['stem']} | {r['index']} | {100*q['iou']:.6f} | {100*q['dice']:.6f} | 0 / 0 |")
lines+=['','All TP/FP/FN, IoU and Dice match both the original StageB and its later reproduction exactly at512 and224; verification.json stores the numbers and file hashes. Labels were used only for display and scoring.','', '## Array paths','']
for name,path in paths.items():lines.append(f'- {name}: [{Path(path).name}]({path})')
lines+=['','## Figures','','![Five columns](actual_predictions_five_columns.png)','','![F1 overlay](actual_predictions_F1_overlay.png)']
(DEST/'README.md').write_text('\n'.join(lines)+'\n');print('VERIFIED: 8 slices; raw grids, hard masks and F1 masks exact; IoU/Dice error0 at224 and512; sealed files unchanged.',flush=True)
