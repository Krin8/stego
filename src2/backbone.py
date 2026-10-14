"""Local-only frozen BiomedCLIP image tokens; no text naming functionality."""
import json
import os
import torch
import torch.nn.functional as F
from .utils import resolve_path, sha256


def load_backbone(cfg):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from open_clip import create_model_and_transforms
    from open_clip.factory import _MODEL_CONFIGS, load_checkpoint
    from timm.layers import Format
    folder = resolve_path(cfg, cfg['paths']['backbone_folder'])
    weights = folder / 'open_clip_pytorch_model.bin'
    if sha256(weights) != cfg['preprocessing']['backbone_sha256']:
        raise ValueError('Backbone weight SHA256 mismatch')
    spec = json.loads((folder / 'open_clip_config.json').read_text())
    for key in ('mean', 'std'):
        if list(spec['preprocess_cfg'][key]) != cfg['preprocessing'][key]:
            raise ValueError('Preprocessing differs from local backbone config')
    name = 'src2_biomedclip_local'
    _MODEL_CONFIGS[name] = spec['model_cfg']
    full, _, _ = create_model_and_transforms(name, pretrained=None, pretrained_text=False, device='cpu')
    loaded = load_checkpoint(full, str(weights), strict=True, weights_only=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError(str(loaded))
    full.eval().requires_grad_(False)
    trunk = full.visual.trunk
    trunk.dynamic_img_size = True
    trunk.patch_embed.strict_img_size = False
    trunk.patch_embed.flatten = False
    trunk.patch_embed.output_fmt = Format.NHWC
    return full


def tokens(full, images):
    full.eval()
    with torch.inference_mode():
        f = full.visual.trunk.get_intermediate_layers(images, n=1, reshape=True, norm=True)[0]
        if f.shape[1:] != (768, 28, 28):
            raise ValueError(f'Wrong dense feature shape: {f.shape}')
        return F.normalize(f, dim=1)
