from common import *
import matplotlib.pyplot as plt
meta=metadata();r=json.loads((OUT/'seed0.json').read_text());variants=['B2_hard','B5_2D','F1','F2']
original=json.loads((OLD/'figure_selection.json').read_text())['eight_indices']
sc=r['variants']['B5_2D']['slice_scores']
eligible=[i for i,s in enumerate(sc) if s['tp']+s['fp']+s['fn']>0]
worst=sorted(eligible,key=lambda i:(sc[i]['iou'],i))[:3];indices=original+worst
assert len(indices)==11
labels=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r')
pred={v:np.load(OUT/f'{v}_s0_pred.npy',mmap_mode='r') for v in variants}
fig,axes=plt.subplots(11,6,figsize=(24,43),squeeze=False);details=[]
for row,i in enumerate(indices):
 rec=meta['records'][i];im=raw_image(rec);truth=labels[i].astype(bool)
 role='original fixed' if row<4 else ('original largest-liver' if row<8 else f'worst B5 rank{row-7}')
 for col,ax in enumerate(axes[row]):
  ax.imshow(im,cmap='gray',vmin=0,vmax=1)
  if col and truth.any():ax.contour(truth,levels=[.5],colors=['red'],linewidths=.8)
  if col<2:ax.set_title(['Input','Ground truth'][col],fontsize=12)
  else:
   v=variants[col-2];mask=pred[v][i].astype(bool)
   if mask.any():ax.contour(mask,levels=[.5],colors=['lime'],linewidths=.8)
   ax.set_title(f"{v} IoU {100*r['variants'][v]['slice_scores'][i]['iou']:.1f}%",fontsize=12)
  ax.set_xticks([]);ax.set_yticks([])
 axes[row,0].set_ylabel(f"Patient {rec['patient']}\n{role}\n{rec['stem']}\nGT pixels={int(truth.sum())}",fontsize=9)
 details.append(dict(index=i,sample_id=rec['sample_id'],patient=rec['patient'],role=role,
  gt_pixels=int(truth.sum()),scores={v:r['variants'][v]['slice_scores'][i] for v in variants}))
fig.suptitle('Fixed F1/F2 refinements | seed0 | native512 | red=GT, green=prediction\nAll prediction mappings inherited from other19 patients; worst/largest-liver slice selection uses labels; neither F1 nor F2 selected',fontsize=15)
fig.tight_layout(rect=[0,0,1,.98]);png=OUT/'eleven_slices_contours.png';pdf=OUT/'eleven_slices_contours.pdf'
fig.savefig(png,dpi=160);fig.savefig(pdf);plt.close(fig)
with Image.open(png) as im:shape=im.size;assert shape[0]>=1600
dump('figures.json',dict(script=__file__,paths=[str(png),str(pdf)],png_size=list(shape),seed=0,
 indices=indices,worst_indices=worst,undefined_iou_excluded=2341-len(eligible),details=details,
 labels_used='Worst-slice selection, truth display and per-slice scores; original largest-liver selection also uses labels'))
print('Saved',shape,png,pdf,flush=True)
