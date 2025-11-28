import torch
import os
import shutil
import numpy as np
import math
import argparse
import time

from diffusers.models import AutoencoderKL
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from accelerate import Accelerator
#from deepspeed.profiling.flops_profiler import FlopsProfiler

from src.utils.utils import init_from_ckpt, find_model
from src.utils.eval_utils import create_npz_from_sample_folder
from src.scheduler.flow_matching import blockwise_flow_matching


def main(args):
    """
    Run sampling.
    """
    if args.precision.tf32:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.capture_scalar_outputs = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP
    accelerator = Accelerator(
        mixed_precision="bf16",
    )
    weight_dtype = torch.float16

    rank = accelerator.process_index
    device = rank % torch.cuda.device_count()
    seed = args.seed.global_seed * accelerator.num_processes + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={accelerator.num_processes}.")

    # Load model:
    BFM_models = find_model(args.model.type)
    latent_size = args.resolution // 8
    block_kwargs = args.model.network_config
    model = BFM_models[args.model.type](
        **block_kwargs
    )

    ckpt_path = args.ckpt_path
    init_from_ckpt(model, ckpt_path, verbose=True)
    if args.compile.enabled:
        model = torch.compile(model)
    model = accelerator.prepare(model).to(device)
    model.eval()  # important!

    if args.profile.enabled:
        flops_profiler = FlopsProfiler(model)

    # Flow Matching scheduler
    scheduler = blockwise_flow_matching(
        prediction=args.sampling.prediction,
        path_type=args.sampling.path_type,
        encoders=None,
        segments=args.model.network_config.segments,
        segment_depth=args.model.network_config.segment_depth,
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema", torch_dtype=weight_dtype).to(device)
    vae.eval()

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.type.replace("/", "-")
    ckpt_string_name = os.path.basename(ckpt_path).replace(".safetensors", "") if ckpt_path else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-" \
                  f"cfg-{args.cfg_scale}-seed-{args.seed.global_seed}-num_steps-{args.sampling.num_steps_per_segment}-{args.sampling.mode}-{args.name}"
    if args.save_dir is not None:
        sample_folder_dir = f"{args.save_dir}/npz/{folder_name}"
    else:
        sample_folder_dir = f"npz/{folder_name}"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.sampling.per_proc_batch_size
    global_batch_size = n * accelerator.num_processes
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.sampling.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, args.model.network_config.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.model.network_config.num_classes, (n,), device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps_per_segment=args.sampling.num_steps_per_segment,
            cfg_scale=args.cfg_scale,
            guidance_low=args.sampling.guidance_low,
            guidance_high=args.sampling.guidance_high,
        )
        with torch.no_grad():

            if args.profile.enabled:
                flops_profiler.start_profile()
                start_time = time.time()

            with accelerator.autocast():
                if args.sampling.mode == "sde":
                    samples = scheduler.sample_sde(**sampling_kwargs).to(weight_dtype)
                elif args.sampling.mode == "ode":
                    samples = scheduler.sample_ode(**sampling_kwargs).to(weight_dtype)
                else:
                    raise NotImplementedError()

                if args.profile.enabled:
                    end_time = time.time()
                    flops_profiler.print_model_profile()
                    total_flops = flops_profiler.get_total_flops()
                    print(f"Time taken: {end_time - start_time} seconds")
                    print(f"Total FLOPs: {total_flops/1e12:.4f} TFLOPs")
                    exit()

                samples = vae.decode(samples / vae.config.scaling_factor).sample

            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * accelerator.num_processes + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    accelerator.wait_for_everyone()
    if rank == 0 and args.save_npz:
        create_npz_from_sample_folder(sample_folder_dir, args.sampling.num_fid_samples)
    accelerator.wait_for_everyone()
    if rank == 0 and args.save_npz:
        print(f"Removing PNG directory: {sample_folder_dir}")
        shutil.rmtree(sample_folder_dir)
        print("PNG directory removed. Done.")
    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--config', type=str)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--save_npz', type=bool, default=False)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.cfg_scale = args.cfg_scale
    config.ckpt_path = args.ckpt_path
    config.save_npz = args.save_npz
    config.save_dir = args.save_dir
    return config

if __name__ == "__main__":
    args = parse_args()
    main(args)
