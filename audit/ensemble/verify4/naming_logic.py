    print("  Naming...")
    sums = np.zeros((8,4), np.float64)
    nearest = [[] for _ in range(8)]
    fallback_count = 0
    yy,xx = np.mgrid[-5:6,-5:6]; disk = xx*xx+yy*yy<=25
    G = 28
    
    for n, i in enumerate(train_ids):
        r = rr[i]
        ds = pydicom.dcmread(r['dicom'])
        hu = ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
        body = ndi.binary_fill_holes(largest(ndi.binary_opening(hu>-500,structure=disk),np.ones((3,3),int)))
        ys,xs = np.where(body)
        y0,x0,side = r['box']
        gy,gx = np.mgrid[:G,:G]; cy=y0+(gy.ravel()+.5)*side/G-.5; cx=x0+(gx.ravel()+.5)*side/G-.5
        iy=np.floor(cy+.5).astype(int); ix=np.floor(cx+.5).astype(int)
        eligible=(iy>=0)&(ix>=0)&(iy<512)&(ix<512); valid=np.flatnonzero(eligible); eligible[valid]=body[iy[valid],ix[valid]]; use=np.flatnonzero(eligible)
        ya=np.clip(np.ceil(y0+gy.ravel()*side/G-.5).astype(int),0,512); yb=np.clip(np.ceil(y0+(gy.ravel()+1)*side/G-.5).astype(int),0,512)
        xa=np.clip(np.ceil(x0+gx.ravel()*side/G-.5).astype(int),0,512); xb=np.clip(np.ceil(x0+(gx.ravel()+1)*side/G-.5).astype(int),0,512)
        def rectsum(a):
            integral=np.pad(a,((1,0),(1,0))).cumsum(0,dtype=np.float64).cumsum(1,dtype=np.float64); return integral[yb,xb]-integral[ya,xb]-integral[yb,xa]+integral[ya,xa]
        count=rectsum(body); total=rectsum(hu*body)
        means=np.divide(total,count,out=np.zeros_like(total),where=count>0); empty=use[count[use]==0]
        means[empty]=hu[iy[empty],ix[empty]]; fallback_count+=len(empty)
        
        z = embedding(i).reshape(-1,70); d_dist = distances(z, centers); assignment = d_dist.argmin(1)
        nx = (cx-xs.min())/max(1,xs.max()-xs.min()); ny = (cy-ys.min())/max(1,ys.max()-ys.min())
        for cluster in range(8):
            u = use[assignment[use]==cluster]
            sums[cluster,0] += len(u)
            sums[cluster,1:] += [nx[u].sum(), ny[u].sum(), means[u].sum()]
            v = d_dist[use,cluster]
            cut = np.partition(v, min(199,len(v)-1))[min(199,len(v)-1)]
            pool = use[v<=cut]
            order = pool[np.lexsort((pool, d_dist[pool,cluster]))][:200]
            candidates = [(float(d_dist[t,cluster]), i, int(t), float(cx[t]), float(cy[t])) for t in order]
            nearest[cluster] = sorted(nearest[cluster]+candidates)[:200]
            
    stats = []
    for k in range(8):
        count = sums[k,0]
        x,y,h = sums[k,1:]/count if count else [float(1),float(1),float(0)]
        stats.append(dict(cluster=k, body_patches=int(count), area=float(count/max(1,sums[:,0].sum())), centroid_x=float(x), centroid_y=float(y), mean_HU=float(h), candidate=bool(count and x<.5 and y<.6 and 40<=h<=180)))
    possible = [x for x in stats if x['candidate']]
    n2 = max(possible, key=lambda x:x['area'])['cluster'] if possible else None
    
    keys=sorted(set((i,t,x,y) for nn in nearest for _,i,t,x,y in nn))
    emb=np.empty((len(keys),512),np.float32)
    for s in range(0,len(keys),16):
        inp=[]
        for i,t,x,y in keys[s:s+16]:
            inp.append(image_tensor(crop_image(load_image(rr[i]),[int(np.floor(y+.5))-56,int(np.floor(x+.5))-56,112]),224))
        with torch.no_grad():
            emb[s:s+len(inp)]=full.encode_image(torch.stack(inp),normalize=True).numpy()
            
    lookup={k:i for i,k in enumerate(keys)}
    avg=np.stack([emb[[lookup[(i,t,x,y)] for _,i,t,x,y in nn]].mean(0) for nn in nearest])
    avg/=np.maximum(np.linalg.norm(avg,axis=1,keepdims=True),1e-12)
    sim=avg@texts.T
    std=sim.std(0,keepdims=True)
    zz=np.divide(sim-sim.mean(0,keepdims=True),std,out=np.zeros_like(sim),where=std>0)
    n1=int(zz[:,0].argmax())
    selected=dict(N1=n1, N2=n2, N3=n1 if n1==n2 else None)
    
