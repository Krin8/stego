"""Build a self-contained F1 utility from exact prior scientific function bodies."""
from pathlib import Path
import ast
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
header='''"""Frozen CT F1 inference. CPU only; no training. See README.md for the PNG contract."""
from pathlib import Path
import os, sys, json, hashlib, tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR=str(HERE/'mpl'))
tempfile.tempdir=str(HERE)
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
sys.path.insert(0,str(ROOT/'src'))
from chaos_preprocessing import MEAN, STD, read_image, file_sha256
from types import SimpleNamespace
torch.set_num_threads(4)
'''
sources={'audit/improve/pipeline.py':['body_box','crop_image','image_tensor','inverse_grid','resize224'], 'audit/improve/postprocess.py':['largest','guided']}
text=header;provenance={}
for rel,names in sources.items():
 p=ROOT/rel;source=p.read_text();tree=ast.parse(source)
 for name in names:
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  body=ast.get_source_segment(source,node);text+='\n# Verbatim from '+rel+'\n'+body+'\n'
  provenance[name]={'source':rel,'line':node.lineno,'sha256':__import__('hashlib').sha256(body.encode()).hexdigest()}
(OUT/'segment_ct.py').write_text(text)
(OUT/'function_provenance.json').write_text(__import__('json').dumps(provenance,indent=2))
