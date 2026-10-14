from common import *
inv=inventory();dump('inventory.json',inv)
(ROOT/'audit/FORENSICS.md').write_text('# CHAOS cluster forensics\n\n## Checkpoints found\n\n'+ '\n'.join(f"- {k}: {'FOUND' if v['exists'] else 'MISSING — skipped'}; `{v['path']}`; {v['bytes']} bytes." for k,v in inv.items())+'\n\nCPU/read-only investigation. Imports actual model/preprocessing/head/probe; each script has570s hard timeout. Only new scratch forensic artifacts and this report may be written. No training, installs, or changes to datasets, checkpoints or pipeline caches.\n')
results={};model=None
for name,item in inv.items():
 if not item['exists']:continue
 model,ck=load(item['path'],model);ds=dataset(model.cfg);assert len(ds)==472
 pred=np.lib.format.open_memmap(OUT/f'{name}_assignments.npy',mode='w+',dtype='uint8',shape=(472,224,224))
 first=not results
 if first:
  featsout=np.lib.format.open_memmap(OUT/'val_features.npy',mode='w+',dtype='float32',shape=(472,768,14,14))
  labels=np.lib.format.open_memmap(OUT/'val_labels.npy',mode='w+',dtype='uint8',shape=(472,224,224))
  dump('val_order.json',[r['sample_id'] for r in ds.dataset.records])
 conf=np.zeros((8,2),dtype=np.int64);monitor=CollapseMonitor(8)
 with torch.inference_mode():
  for batch in DataLoader(ds,batch_size=16,shuffle=False,num_workers=0):
   feats,code=model.net(batch['img']);up=F.interpolate(code,batch['label'].shape[-2:],mode='bilinear',align_corners=False)
   _,prob=model.cluster_probe(up,None);ids=prob.argmax(1);monitor.update(code,ids);model.cluster_metrics.update(ids,batch['label'])
   idx=batch['ind'].numpy();pr=ids.numpy();y=batch['label'].numpy();assert set(np.unique(y))<={0,1}
   pred[idx]=pr;conf+=np.bincount((pr*2+y).ravel(),minlength=16).reshape(8,2)
   if first:featsout[idx]=feats.numpy();labels[idx]=y
  actual=model.cluster_metrics.compute()
 r=mapping_report(conf);assert abs(r['majority']['iou']*100-float(actual['test/cluster/liver_IoU']))<1e-5
 r.update(path=item['path'],batch=ck['global_step']//3,monitors=monitor.compute(),script=__file__,elapsed_seconds=elapsed());results[name]=r
 dump('task1.json',results);pred.flush()
 if first:featsout.flush();labels.flush()
 note(f"Task1 {name}: majority IoU{100*r['majority']['iou']:.6f}%, optimal-subset IoU{100*r['optimal']['iou']:.6f}%, best single cluster{r['best_single']['clusters']}. Producer`audit/scratch/forensics/task1.py`; detail`task1.json`. Labels used only for scoring. Runtime{elapsed():.2f}s so far.")
 print(name,'majority',r['majority']['iou'],'optimal',r['optimal']['iou'],'single',r['best_single'],flush=True)
 del ck
print('DONE',elapsed(),flush=True)
