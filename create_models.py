"""Functions to create models."""

from topodiff.unet import (
    UNetModel,
    EncoderUNetModel,
)
from topodiff import gaussian_diffusion as gd
from topodiff.respace import SpacedDiffusion
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

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

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
