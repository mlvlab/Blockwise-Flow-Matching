import torch
import torch.nn as nn
import torch.cuda.amp as amp

from functools import partial
from typing import Optional
from einops import rearrange, repeat
from src.model.modules import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder,
    TransformerBlock, FinalLayer
)
from src.model.utils import make_grid_mask_size, get_parameter_dtype
from src.utils.utils import init_from_ckpt
from src.model.rope import VisionRotaryEmbedding

class BFM(nn.Module):
    def __init__(
        self,
        context_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = False,
        use_checkpoint: bool=False,
        use_swiglu: bool = False,
        use_swiglu_large: bool = False,
        rel_pos_embed: Optional[str] = 'rope',
        norm_type: str = "layernorm",
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        rope_theta: float = 10000.0,
        custom_freqs: str = 'normal',
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        online_rope: bool = False,
        add_rel_pe_to_v: bool = False,
        time_shifting: int = 1,
        max_cached_len: int = 256,
        concat_adaln: bool = False,
        # for BFM training
        segments: int = 6,
        segment_depth: int = 5,
        feature_alignment_depth: int = 20,
        dim_projection: int = 768,
        adaln_bias: bool = True,
        adaln_type: str = "lora",
        adaln_lora_dim: int = 288,
        # for finetuning
        ignore_keys: list = None,
        finetune: bool = False,
        frn_depth: int = 4,
        pretrain_ckpt: str = None,
        **kwargs,
    ):
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = self.in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.adaln_type = adaln_type
        self.online_rope = online_rope
        self.time_shifting = time_shifting

        # for BFM training
        self.sigmas = torch.linspace(0, 1, segments+1)
        self.segments = segments
        self.segment_depth = segment_depth
        self.feature_alignment_depth = feature_alignment_depth

        # Rotary embedding
        self.rel_pos_embed = VisionRotaryEmbedding(
            head_dim=hidden_size//num_heads, theta=rope_theta, custom_freqs=custom_freqs, online_rope=online_rope,
            max_pe_len_h=max_pe_len_h, max_pe_len_w=max_pe_len_w, decouple=decouple, ori_max_pe_len=ori_max_pe_len,
            max_cached_len=max_cached_len,
        )

        # Condition embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Feature alignment network
        self.representation_x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.representation_blocks = nn.ModuleList([TransformerBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim, feature_alignment=True
        ) for _ in range(feature_alignment_depth)])

        # Feature projection network
        self.linear_projection = nn.Sequential(
                nn.Linear(hidden_size, 2048),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.SiLU(),
                nn.Linear(2048, dim_projection),
            )

        # Velocity network
        self.x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.velocity_blocks = nn.ModuleList([nn.ModuleList([TransformerBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim, concat_adaln=concat_adaln
        ) for _ in range(segment_depth)]) for _ in range(segments)])

        # AdaLN modulation
        if adaln_type == 'lora':
            self.global_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias)
            )
            self.global_adaLN_modulation2 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size * 2, 6 * hidden_size, bias=adaln_bias)
            )
        else:
            self.global_adaLN_modulation = None
            self.global_adaLN_modulation2 = None

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type, concat_adaln=True)

        self.finetune = finetune
        if finetune:
            self.frn_blocks = nn.ModuleList([TransformerBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type='normal', adaln_lora_dim=adaln_lora_dim, feature_alignment=True
            ) for _ in range(frn_depth)])
            self.freeze_parameters(unfreeze=['frn_blocks'])

        self.initialize_weights(pretrain_ckpt=pretrain_ckpt, ignore=ignore_keys)

    def initialize_weights(self, pretrain_ckpt=None, ignore=None):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.representation_x_embedder.proj.weight.data)
        nn.init.constant_(self.representation_x_embedder.proj.bias, 0)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize condition embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for blocks in self.velocity_blocks:
            for block in blocks:
                if self.adaln_type in ['normal', 'lora']:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                elif self.adaln_type == 'swiglu':
                    nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                    nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)

        # Zero-out adaLN modulation layers in representation blocks:
        for block in self.representation_blocks:
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)

        # Zero-out adaLN modulation layers in global AdaLN blocks:
        if self.adaln_type == 'lora':
            nn.init.constant_(self.global_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.global_adaLN_modulation2[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation2[-1].bias, 0)

        # Zero-out adaLN modulation layers in final layer:
        if self.adaln_type == 'swiglu':
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.bias, 0)
        else:   # adaln_type in ['normal', 'lora']
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers in final layer:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        keys = list(self.state_dict().keys())
        ignore_keys = []
        if ignore != None:
            for ign in ignore:
                for key in keys:
                    if ign in key:
                        ignore_keys.append(key)
        ignore_keys = list(set(ignore_keys))
        if pretrain_ckpt != None:
            init_from_ckpt(self, pretrain_ckpt, ignore_keys, verbose=True)

    def flatten(self, x):
        x = x.reshape(x.shape[0], -1, self.context_size//self.patch_size, self.patch_size, self.context_size//self.patch_size, self.patch_size)
        x = rearrange(x, 'b c h1 p1 h2 p2 -> b (c p1 p2) (h1 h2)')
        x = x.permute(0, 2, 1)  # (b, h, c)
        return x

    def unpatchify(self, x):
        """
        args:
            x: (B, p**2 * C_out, N)
            N = h//p * w//p
        return:
            imgs: (B, C_out, H, W)
        """
        h = w = self.context_size
        p = self.patch_size
        x = rearrange(x, "b (h w) c -> b h w c", h=h//p, w=w//p) # (B, h//2 * w//2, 16) -> (B, h//2, w//2, 16)
        x = rearrange(x, "b h w (c p1 p2) -> b c (h p1) (w p2)", p1=p, p2=p) # (B, h//2, w//2, 16) -> (B, h, w, 4)
        return x

    def forward(self,
                x,
                t,
                y,
                segment_idx=None,
                split_sizes=None,
                t_start=None,
                x_start=None,
                coeff=None,
                representation_feature=None
                ):
        if not self.training:
            return self.forward_sample(x, t, y, segment_idx, representation_feature, coeff)

        if self.finetune:
            return self.forward_ft(x, t, y, segment_idx, split_sizes, t_start, x_start, coeff)

        return self.forward_train(x, t, y, segment_idx, split_sizes)

    def forward_train(self,
                x,
                t,
                y,
                segment_idx=None,
                split_sizes=None):

        x = self.flatten(x)
        with amp.autocast(dtype=torch.float32):
            grid, _, _ = make_grid_mask_size(x.shape[0], self.context_size // self.patch_size, self.context_size // self.patch_size, self.patch_size, x.device)
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
            t = self.t_embedder(t)
            y = self.y_embedder(y, self.training)           # (B, D)
            c = t + y

        if self.global_adaLN_modulation != None:
            global_adaln = self.global_adaLN_modulation(c)
        else:
            global_adaln = 0.0

        # Feature alignment
        representation_noise = self.representation_x_embedder(x)
        for rep_block in self.representation_blocks:
            if not self.use_checkpoint:
                representation_noise = rep_block(representation_noise, c, None, freqs_cos, freqs_sin, global_adaln)
            else:
                representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, None, freqs_cos, freqs_sin, global_adaln)

        # Feature projection
        representation_linear = self.linear_projection(representation_noise)
        c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)
        # c_repre = torch.cat([t.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

        if self.global_adaLN_modulation != None:
            global_adaln2 = self.global_adaLN_modulation2(c_repre)
        else:
            global_adaln2 = 0.0

        x = self.x_embedder(x)

        x_splits = torch.split(x, split_sizes, dim=0)
        c_repre_splits = torch.split(c_repre, split_sizes, dim=0)
        freqs_cos_splits = torch.split(freqs_cos, split_sizes, dim=0)
        freqs_sin_splits = torch.split(freqs_sin, split_sizes, dim=0)
        global_adaln2_splits = torch.split(global_adaln2, split_sizes, dim=0)
        outputs = []

        for seg_idx, split_size in enumerate(split_sizes):
            if split_size == 0:
                continue
            x_split_idx = x_splits[seg_idx]
            c_repre_split_idx = c_repre_splits[seg_idx]
            freqs_cos_split_idx = freqs_cos_splits[seg_idx]
            freqs_sin_split_idx = freqs_sin_splits[seg_idx]
            global_adaln2_split_idx = global_adaln2_splits[seg_idx]

            for blocks in self.velocity_blocks[seg_idx]:
                if not self.use_checkpoint:
                    x_split_idx = blocks(x_split_idx, c_repre_split_idx, None, freqs_cos_split_idx, freqs_sin_split_idx, global_adaln2_split_idx)
                else:
                    x_split_idx = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(blocks), x_split_idx, c_repre_split_idx, None, freqs_cos_split_idx, freqs_sin_split_idx, global_adaln2_split_idx)

            x_split_idx = self.final_layer(x_split_idx, c_repre_split_idx)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
            outputs.append(x_split_idx)

        x = torch.cat(outputs, dim=0)
        x = self.unpatchify(x)
        return x, representation_linear

    def forward_ft(
        self,
        x,
        t,
        y,
        segment_idx=None,
        split_sizes=None,
        t_start=None,
        x_start=None,
        coeff=None
        ):

        with torch.no_grad():
            x = self.flatten(x)
            x_start = self.flatten(x_start)

            grid, _, _ = make_grid_mask_size(x.shape[0], self.context_size // self.patch_size, self.context_size // self.patch_size, self.patch_size, x.device)
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

            t = t.float().to(x.dtype)
            t = self.t_embedder(t)
            y = self.y_embedder(y, True)           # (B, D)
            c = t + y

            t_start = t_start.float().to(x.dtype)
            t_start = self.t_embedder(t_start)
            c_start = t_start + y

            if self.global_adaLN_modulation != None:
                global_adaln_start = self.global_adaLN_modulation(c_start)
                global_adaln = self.global_adaLN_modulation(c)
            else:
                global_adaln_start = 0.0
                global_adaln = 0.0

            # Feature alignment
            representation_noise_start = self.representation_x_embedder(x_start)
            for rep_block in self.representation_blocks:
                if not self.use_checkpoint:
                    representation_noise_start = rep_block(representation_noise_start, c_start, None, freqs_cos, freqs_sin, global_adaln_start)
                else:
                    representation_noise_start = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise_start, c_start, None, freqs_cos, freqs_sin, global_adaln_start)

            representation_noise = self.representation_x_embedder(x)
            representation_noise_clone = representation_noise.detach().clone()
            for rep_block in self.representation_blocks:
                if not self.use_checkpoint:
                    representation_noise = rep_block(representation_noise, c, None, freqs_cos, freqs_sin, global_adaln)
                else:
                    representation_noise = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(rep_block), representation_noise, c, None, freqs_cos, freqs_sin, global_adaln)

        ### Residual Approximation ###
        approx = representation_noise_start.detach().clone()
        for frn_block in self.frn_blocks:
            if not self.use_checkpoint:
                representation_noise_clone = frn_block(representation_noise_clone, c, None, freqs_cos, freqs_sin, 0.0)
            else:
                representation_noise_clone = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(frn_block), representation_noise_clone, c, None, freqs_cos, freqs_sin, 0.0)
        approx = approx + coeff * representation_noise_clone

        return approx, representation_noise

    @torch.no_grad()
    def forward_sample(
        self,
        x,
        t,
        y,
        segment_idx=None,
        representation_feature=None,
        coeff=1.0,
        ):

        x = self.flatten(x)
        with amp.autocast(dtype=torch.float32):
            grid, _, _ = make_grid_mask_size(x.shape[0], self.context_size // self.patch_size, self.context_size // self.patch_size, self.patch_size, x.device)
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

            t = self.t_embedder(t)
            y = self.y_embedder(y, self.training)           # (B, D)
            c = t + y

        if self.global_adaLN_modulation != None:
            global_adaln = self.global_adaLN_modulation(c)
        else:
            global_adaln = 0.0

        # Feature alignment
        if representation_feature is None:
            representation_noise = self.representation_x_embedder(x)
            for rep_block in self.representation_blocks:
                representation_noise = rep_block(representation_noise, c, None, freqs_cos, freqs_sin, global_adaln)
            approx = representation_noise.detach().clone()
        else:
            approx = representation_feature.detach().clone()
            representation_noise = self.representation_x_embedder(x)
            for frn_block in self.frn_blocks:
                representation_noise = frn_block(representation_noise, c, None, freqs_cos, freqs_sin, 0.0)
            coeff = coeff.float().to(x.dtype)
            representation_noise = approx + coeff * representation_noise

        c_repre = torch.cat([c.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)
        # c_repre = torch.cat([t.unsqueeze(1).repeat(1, representation_noise.shape[1], 1), representation_noise], dim=-1)

        if self.global_adaLN_modulation != None:
            global_adaln2 = self.global_adaLN_modulation2(c_repre)
        else:
            global_adaln2 = 0.0

        x = self.x_embedder(x)

        for blocks in self.velocity_blocks[segment_idx]:
            x = blocks(x, c_repre, None, freqs_cos, freqs_sin, global_adaln2)

        x = self.final_layer(x, c_repre)                      # (B, N, p ** 2 * C_out), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        x = self.unpatchify(x)
        return x, approx

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward


    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


    def freeze_parameters(self, unfreeze):
        for name, param in self.named_parameters():
                param.requires_grad = False
        for unf in unfreeze:
            for name, param in self.named_parameters():
                if unf in name: # LN means Layer Norm
                    param.requires_grad = True


#################################################################################
#                                   SiT Configs                                  #
#################################################################################
def BFM_XL(**kwargs):
    return BFM(hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

BFM_models = {
    "BFM_XL": BFM_XL,
}