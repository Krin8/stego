"""Exercise the actual entrypoint seed call without constructing data/models."""
import ast
from pathlib import Path
from omegaconf import OmegaConf
root=Path(__file__).resolve().parents[2]
cfg=OmegaConf.load(root/'src/configs/train_config.yaml')
module=ast.parse((root/'src/train_segmentation.py').read_text())
main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='my_app')
node=next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='seed_everything')
for requested in [0,1,2]:
 cfg.seed=requested;seen=[]
 eval(compile(ast.Expression(node),'<actual entrypoint seed call>','eval'),{'cfg':cfg,'seed_everything':lambda **kw:seen.append(kw['seed'])})
 print(f'requested={requested} observed={seen[0]}')
 assert seen==[requested], 'Configured seed ignored'
assert OmegaConf.load(root/'src/configs/train_config.yaml').seed==0
print('PASS: default0, overrides1/2 reach actual seed call')
