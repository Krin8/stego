"""Fixed eight-slice PCA diagnostic; no labels fit PCA or set colours."""
from common import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
os.environ['LABEL_FREE']='1'
rr=records();inds=json.loads((OUT/'PROTOCOL.json').read_text())['figures']['indices'];f=np.load(OUT/'D1_features.npy',mmap_mode='r')
x=np.array(f[inds]).reshape(-1,768);pca=PCA(n_components=3,svd_solver='full');z=pca.fit_transform(x)
lo,hi=np.percentile(z,[1,99],axis=0);rgb=np.clip((z-lo)/(hi-lo),0,1).reshape(8,28,28,3)
os.environ['LABEL_FREE']='0';truth=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r')
fig,axes=plt.subplots(8,3,figsize=(12,30))
for row,i in enumerate(inds):
 im=raw_image(rr[i]);axes[row,0].imshow(im,cmap='gray',vmin=0,vmax=1);axes[row,0].set_title(rr[i]['stem'])
 axes[row,1].imshow(im,cmap='gray',vmin=0,vmax=1);axes[row,1].contour(truth[i],levels=[.5],colors=['red'],linewidths=.7);axes[row,1].set_title('Ground truth (display only)')
 axes[row,2].imshow(rgb[row],interpolation='nearest');axes[row,2].set_title('D1 PCA RGB — body crop, 28×28')
 for ax in axes[row]:ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'D1_eight_slices_PCA.png',dpi=180);plt.close(fig)
dump('figures.json',dict(producer='figures.py',indices=inds,path=str(OUT/'D1_eight_slices_PCA.png'),PCA_labels_used=False,GT_labels='display only; inherited fixed largest-liver slice selection used labels historically',explained_variance_ratio=pca.explained_variance_ratio_.tolist(),scale_percentiles=[1,99]))
