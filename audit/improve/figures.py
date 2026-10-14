"""Eight pre-declared validation slices, seed0, native-frame contours."""
from pipeline import *
import matplotlib.pyplot as plt
meta=metadata();summary=json.loads((OUT/'summary.json').read_text());best=summary['best'];assert best
selection=json.loads((OUT/'figure_selection.json').read_text());indices=selection['eight_indices']
labels=np.load(OUT/'scoring_masks_native.npy',mmap_mode='r')
basegrids=np.load(OUT/'B0_s0_grids.npy',mmap_mode='r');base=json.loads((OUT/'B0_s0_scores.json').read_text())
bestresult=json.loads((OUT/f'{best}_s0_scores.json').read_text())
if best.startswith(('B4','B5')):post=np.load(OUT/'postprocess_figure_predictions.npz')
else:bestgrids=np.load(OUT/f'{best}_s0_grids.npy',mmap_mode='r')
fig,axes=plt.subplots(8,4,figsize=(16,29),squeeze=False);details=[]
for row,i in enumerate(indices):
 r=meta['records'][i];image=raw_image(r);truth=np.array(labels[i],dtype=bool);patient=r['patient']
 lookup=np.zeros(9,bool);lookup[base['results']['512']['folds'][patient]['optimal']['clusters']]=True
 b0=lookup[native_prediction(basegrids[i],r,'B0').astype(np.uint8)]
 if best.startswith(('B4','B5')):prediction=post[f'{best}_512_optimal_{i}'].astype(bool)
 else:
  lookup=np.zeros(9,bool);lookup[bestresult['results']['512']['folds'][patient]['optimal']['clusters']]=True
  prediction=lookup[native_prediction(bestgrids[i],r,best).astype(np.uint8)]
 bs=scores(confusion(b0,truth,2),[1]);vs=scores(confusion(prediction,truth,2),[1])
 titles=['Input', 'Ground truth',f"B0 IoU {100*bs['iou']:.1f}%",f"{best} IoU {100*vs['iou']:.1f}%"]
 for col,ax in enumerate(axes[row]):
  ax.imshow(image,cmap='gray',vmin=0,vmax=1)
  if col in [1,2,3] and truth.any():ax.contour(truth,levels=[.5],colors=['red'],linewidths=.8)
  if col in [2,3]:
   pred=b0 if col==2 else prediction
   if pred.any():ax.contour(pred,levels=[.5],colors=['lime'],linewidths=.8)
  ax.set_title(titles[col],fontsize=11);ax.set_xticks([]);ax.set_yticks([])
 role='fixed midpoint' if row<4 else 'largest liver (labels)'
 axes[row,0].set_ylabel(f"Patient {patient}\n{role}\n{r['stem']}",fontsize=8)
 details.append(dict(index=i,sample_id=r['sample_id'],patient=patient,role=role,B0=bs,best=vs))
fig.suptitle(f'Frozen CT clustering | seed0 | native512 LOPO maps | red=truth, green=prediction\nBest={best}; best selection and largest-liver slice selection use cohort labels',fontsize=12)
fig.tight_layout(rect=[0,0,1,.975]);path=OUT/'eight_slices_contours.png';fig.savefig(path,dpi=160);plt.close(fig)
dump('figures.json',dict(script=__file__,paths=[str(path)],seed=0,best=best,details=details,
 labels_used='Native heldout labels for per-slice scoring and truth contours; other19 labels for mapping; largest-liver and best selection use labels'))
print(path,flush=True)
