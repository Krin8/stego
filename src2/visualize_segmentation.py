"""Render actual saved cluster/hard/refined arrays; never refit during plotting."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
from src2.utils import parser,load_config,resolve_path,new_output,read_json,sha256
from src2.preprocessing import load_image,inverse_grid


def visualize(cfg, evaluation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    evaluation=Path(evaluation);manifest=read_json(evaluation/'prediction_manifest.json')
    for name,h in manifest['outputs'].items():
        if sha256(evaluation/name)!=h:raise ValueError('Saved predictions do not match seal')
    out=new_output(output)
    raw=np.load(evaluation/'raw_clusters.npy',mmap_mode='r');hard=np.load(evaluation/'N2_hard.npy',mmap_mode='r');final=np.load(evaluation/'N2_refined.npy',mmap_mode='r')
    masks=np.load(resolve_path(cfg,cfg['paths']['scoring_masks']),mmap_mode='r')
    ids=cfg['fixed_indices'];lookup={i:j for j,i in enumerate(manifest['indices'])}
    if any(i not in lookup for i in ids):raise ValueError('Fixed visualization IDs absent from evaluation')
    cmap=ListedColormap([*plt.get_cmap('tab10').colors[:8],(0.,0.,0.)]);norm=BoundaryNorm(np.arange(-.5,9.5),9)
    fig,axes=plt.subplots(len(ids),5,figsize=(16,3.2*len(ids)),squeeze=False)
    for row,i in enumerate(ids):
        j=lookup[i];r=manifest['records'][j];im=load_image(resolve_path(cfg,r['image']),cfg)
        native=inverse_grid(raw[j],r,mode='nearest',outside=8)
        panels=[im,masks[i]==1,native,hard[j],final[j]]
        for col,panel in enumerate(panels):
            ax=axes[row,col]
            if col==2:artist=ax.imshow(panel,cmap=cmap,norm=norm,interpolation='nearest')
            else:ax.imshow(panel,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            ax.set_axis_off()
        axes[row,0].set_title(r['stem'])
    for col,title in enumerate(['CT','Ground truth','Raw K=8 clusters','N2 hard mask','N2 guided refinement']):
        axes[0,col].set_title(title+'\n'+(manifest['records'][lookup[ids[0]]]['stem'] if col==0 else ''))
    fig.suptitle(f"Optimized P1 seed 0 — N2 liver cluster {manifest['selected_cluster']}; black outside crop",fontsize=13)
    bar=fig.colorbar(artist,ax=axes[:,2].tolist(),ticks=np.arange(9),fraction=.035)
    bar.ax.set_yticklabels([f'C{i}' for i in range(8)]+['outside'])
    fig.savefig(out/'fixed_slices.png',dpi=160,bbox_inches='tight');fig.savefig(out/'fixed_slices.pdf',bbox_inches='tight');plt.close(fig)
    print(out/'fixed_slices.png')


def main():
    p=parser('plot_config.yaml',__doc__);p.add_argument('--evaluation-dir');p.add_argument('--output')
    a=p.parse_args();cfg=load_config(a.config)
    visualize(cfg,a.evaluation_dir or resolve_path(cfg,cfg['evaluation_dir']),a.output or resolve_path(cfg,cfg['output']))


if __name__=='__main__':main()
