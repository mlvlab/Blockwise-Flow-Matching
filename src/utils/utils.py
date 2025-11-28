import torch
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from collections import OrderedDict
from safetensors.torch import load_file, save_file
import re


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def preprocess_raw_image(x, enc_type, resolution=256):
    if 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    else:
        raise NotImplementedError
    return x

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        name = name.replace("_orig_mod.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512) # currently only support 256x256 and 512x512

    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        architectures.append(architecture)
        encoder_types.append(encoder_type)

        if 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        encoders.append(encoder)
    return encoders, encoder_types, architectures

def find_model(arch):
    if arch == 'BFM_XL':
        from src.model.bfm import BFM_models
    elif arch == 'SiT':
        from src.models.sit import SiT_models as BFM_models
    else:
        raise NotImplementedError
    return BFM_models

def init_from_ckpt(
    model, checkpoint_dir, ignore_keys=None, verbose=False
) -> None:
    if checkpoint_dir.endswith(".safetensors"):
        try:
            model_state_dict=load_file(checkpoint_dir)
        except:
            model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    else:
        model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    model_new_ckpt=dict()
    for i in model_state_dict.keys():
        model_new_ckpt[i] = model_state_dict[i]

    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(model_new_ckpt.keys())

    # Handle _orig_mod prefix mismatches from torch.compile
    if checkpoint_keys != model_keys:
        # Check if model has _orig_mod prefix but checkpoint doesn't
        model_has_orig_mod = any(k.startswith('_orig_mod.') for k in model_keys)
        checkpoint_has_orig_mod = any(k.startswith('_orig_mod.') for k in checkpoint_keys)

        if model_has_orig_mod and not checkpoint_has_orig_mod:
            # Add _orig_mod prefix to checkpoint keys
            new_model_new_ckpt = {}
            for k, v in model_new_ckpt.items():
                new_key = f'_orig_mod.{k}'
                new_model_new_ckpt[new_key] = v
            model_new_ckpt = new_model_new_ckpt
            if verbose:
                print("Added '_orig_mod.' prefix to checkpoint keys to match compiled model.")

        elif not model_has_orig_mod and checkpoint_has_orig_mod:
            # Remove _orig_mod prefix from checkpoint keys
            new_model_new_ckpt = {}
            for k, v in model_new_ckpt.items():
                if k.startswith('_orig_mod.'):
                    new_key = k[len('_orig_mod.'):]
                    new_model_new_ckpt[new_key] = v
                else:
                    new_model_new_ckpt[k] = v
            model_new_ckpt = new_model_new_ckpt
            if verbose:
                print("Removed '_orig_mod.' prefix from checkpoint keys to match non-compiled model.")

    keys = list(model_new_ckpt.keys())
    for k in keys:
        if ignore_keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print("Deleting key {} from state_dict.".format(k))
                    del model_new_ckpt[k]
    missing, unexpected = model.load_state_dict(model_new_ckpt, strict=True)
    if verbose:
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    if verbose:
        print("")

# save and load checkpoint
