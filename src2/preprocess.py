"""Plan or create windowed uint16 CT PNGs from one flat input folder."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
from PIL import Image
from src2.utils import parser,load_config,new_output,write_json,contract
from src2.preprocessing import load_image,body_box


def main():
    p=parser('pipeline.yaml',__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',type=Path);p.add_argument('--write',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    paths=sorted(x for x in a.input.iterdir() if x.is_file() and x.suffix.lower() in ('.dcm','.png'))
    if not paths or len({x.stem for x in paths})!=len(paths):raise ValueError('Expected uniquely named CT slices')
    if a.write and not a.output:p.error('--write requires --output')
    out=new_output(a.output) if a.write else None;records=[]
    for path in paths:
        image=load_image(path,cfg);record=dict(source=str(path.resolve()),shape=list(image.shape),box=body_box(image))
        if out:
            name=path.stem+'.png';Image.fromarray(np.rint(image*65535).astype(np.uint16)).save(out/name);record['output']=name
        records.append(record)
    if out:write_json(out/'preprocessing.json',dict(contract=contract(cfg),records=records))
    print(dict(slices=len(records),written=bool(out),output=str(out) if out else None))


if __name__=='__main__':main()
