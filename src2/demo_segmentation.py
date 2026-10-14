"""Predict new CT PNG/DICOM slices with a saved train-only N2 clustering fit."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
from PIL import Image
from src2.utils import parser, load_config, setup, resolve_path, new_output, write_json, labels_closed, contract, sha256
from src2.modules import load_head
from src2.backbone import load_backbone, tokens
from src2.preprocessing import load_image, body_box, crop_image, image_tensor, inverse_grid
from src2.clustering import load_fit, embed, fields
from src2.postprocessing import refine


def save_overlay(path, image, mask):
    rgb = np.repeat(image[...,None], 3, axis=2)
    rgb[mask] = .65*rgb[mask]+.35*np.array([1.,0.,0.])
    Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(path)


def predict(cfg, folder, fit_dir, output):
    setup(cfg)
    with labels_closed():
        paths = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ('.dcm','.png'))
        if not paths or len({p.stem for p in paths}) != len(paths):
            raise ValueError('Input must contain uniquely named CT .dcm/.png slices, without nested folders')
        model, head_sha = load_head(cfg); centers, fitted = load_fit(fit_dir,cfg,head_sha)
        full = load_backbone(cfg); out = new_output(output)
        selected = fitted['naming']['selected_cluster']; records = []
        for p in paths:
            image = load_image(p,cfg); box = body_box(image)
            record = dict(box=box,shape=list(image.shape))
            x = image_tensor(crop_image(image,box),cfg)[None]
            z = embed(model,tokens(full,x))[0]
            raw, field = fields(z,centers,fitted['temperature'],selected)
            native = inverse_grid(raw,record,mode='nearest',outside=8).astype(np.uint8)
            hard = native==selected if selected is not None else np.zeros(image.shape,bool)
            mask = refine(field,image,record,cfg)
            np.save(out/f'{p.stem}_clusters.npy',raw,allow_pickle=False)
            np.save(out/f'{p.stem}_clusters_native.npy',native,allow_pickle=False)
            Image.fromarray(hard.astype(np.uint8)*255).save(out/f'{p.stem}_hard.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(out/f'{p.stem}_mask.png')
            save_overlay(out/f'{p.stem}_overlay.png',image,mask)
            records.append(dict(input=str(p.resolve()),input_sha256=sha256(p),**record))
        write_json(out/'manifest.json',dict(contract=contract(cfg),checkpoint_sha256=head_sha,
            fit_sha256=sha256(Path(fit_dir)/'fit.json'),selected_cluster=selected,abstained=selected is None,
            records=records,labels_read=False,outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
        print(out)


def main():
    p=parser('demo_config.yaml',__doc__);p.add_argument('--input',required=True);p.add_argument('--fit-dir');p.add_argument('--output')
    a=p.parse_args();cfg=load_config(a.config)
    predict(cfg,a.input,a.fit_dir or resolve_path(cfg,cfg['fit_dir']),a.output or resolve_path(cfg,cfg['output']))


if __name__=='__main__':
    main()
