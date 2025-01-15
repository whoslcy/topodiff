"""Functions to create models."""

from topodiff.unet import (
    UNetModel,
    EncoderUNetModel,
)
from topodiff import gaussian_diffusion as gd
from topodiff.respace import SpacedDiffusion, space_timesteps
from typing import List
import math
import numpy as np

image_size = 64

if image_size == 512:
    channel_multiplier = (0.5, 1, 1, 2, 2, 4, 4)
elif image_size == 256:
    channel_multiplier = (1, 1, 2, 2, 4, 4)
elif image_size == 128:
    channel_multiplier = (1, 1, 2, 3, 4)
elif image_size == 64:
    channel_multiplier = (1, 2, 3, 4)
else:
    raise ValueError(f"unsupported image size: {image_size}")


def get_attention_resolutions(attention_resolutions: List):
    return tuple(image_size // resolution for resolution in attention_resolutions)


def classifier():
    return EncoderUNetModel(
        image_size=image_size,
        in_channels=1,
        model_channels=128,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=get_attention_resolutions([32, 16, 8]),
        channel_mult=channel_multiplier,
        use_fp16=False,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        pool="attention",
    )


def regressor():
    return EncoderUNetModel(
        image_size=image_size,
        in_channels=8,
        model_channels=128,
        out_channels=1,
        num_res_blocks=4,
        attention_resolutions=get_attention_resolutions([32, 16, 8]),
        channel_mult=channel_multiplier,
        use_fp16=False,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        pool="spatial",
    )

steps = 1000

def alpha_bar(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
betas = []
for i in range(steps):
    t1 = i / steps
    t2 = (i + 1) / steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
betas = np.array(betas)

def spaced_diffusion():
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, [100]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )
    

def gaussian_diffusion(timestep_respacing: str = ""):
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


def mean_variance():
    return UNetModel(
        image_size=image_size,
        in_channels=6,
        model_channels=128,
        out_channels=2,
        num_res_blocks=3,
        attention_resolutions=get_attention_resolutions([16, 8]),
        dropout=0.3,
        channel_mult=channel_multiplier,
        use_fp16=True,
        num_heads=4,
        use_scale_shift_norm=True,
    )
