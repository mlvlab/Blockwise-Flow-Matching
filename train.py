import os
import torch
import argparse
import torchvision
import logging
import shutil
import numpy as np
from pathlib import Path
import torch.distributed as dist

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, TorchDynamoPlugin
from tqdm.auto import tqdm
from diffusers.models import AutoencoderKL
from copy import deepcopy
from torch.utils.data import DataLoader

from src.utils.utils import find_model, load_encoders, requires_grad, update_ema, preprocess_raw_image, init_from_ckpt
from src.utils.logging_utils import create_logger, setup_terminal_logging
from src.utils.eval_utils import calculate_inception_stats_imagenet
from src.utils.evaluator import Evaluator
from src.utils.parallelize import apply_compile
from src.data.dataset import CustomDataset
from src.scheduler.flow_matching import blockwise_flow_matching

os.environ["TMPDIR"] = "/nfs/dogyun/data/tmpdir"
os.environ["TEMP"] = "/nfs/dogyun/data/tmpdir"
os.environ["TMP"] = "/nfs/dogyun/data/tmpdir"
os.environ["TORCH_HOME"] = "/nfs/dogyun/data/torch"
os.environ["HF_HOME"] = "/nfs/dogyun/data/huggingface"


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    logging_dir = Path(args.logging.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.logging.output_dir, logging_dir=logging_dir
        )

    dynamo_plugin = None

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.optimization.gradient_accumulation_steps,
        mixed_precision=args.optimization.mixed_precision,
        log_with=args.logging.report_to if args.logging.report_to != 'none' else None,
        project_config=accelerator_project_config,
        dynamo_plugin=dynamo_plugin,
    )

    if args.optimization.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    elif args.optimization.mixed_precision == 'fp32':
        weight_dtype = torch.float32

    save_dir = os.path.join(args.logging.output_dir, args.logging.exp_name)
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    gen_results_dir = f"{save_dir}/gen_results"  # Stores generated samples
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(args.logging.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(gen_results_dir, exist_ok=True)

    if accelerator.is_main_process:
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        logger.info(f"Checkpoint directory created at {checkpoint_dir}")
        logger.info(f"Generated results directory created at {gen_results_dir}")

    if accelerator.is_main_process:
        setup_terminal_logging(save_dir, int(os.environ.get("RANK", 0)))
        OmegaConf.save(config=args, f=os.path.join(save_dir, "config.yaml"))
        logger.info(f"Config saved to {os.path.join(save_dir, 'config.yaml')}")

        # init wandb
        project_name = getattr(args.logging, 'BFM', args.logging.project_name)
        accelerator.init_trackers(
            project_name=project_name,
            init_kwargs={
                "wandb": {"name": f"{args.logging.exp_name}"}
            },
        )
        logger.info(f"Wandb initialized")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.optimization.seed is not None:
        set_seed(args.optimization.seed + accelerator.process_index)

    # set allow_tf32
    if args.optimization.allow_tf32:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.capture_scalar_outputs = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create model:
    assert args.dataset.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.dataset.resolution // 8

    if args.loss.enc_type != "None":
        encoders, encoder_types, architectures = load_encoders(
            args.loss.enc_type, device, args.dataset.resolution
            )
    else:
        encoders, encoder_type, architectures = None, None, None
    dim_projection = [encoder.embed_dim for encoder in encoders] if args.loss.enc_type != 'None' else [0]

    BFM_models = find_model(args.model.type)
    block_kwargs = args.model.network_config
    model = BFM_models[args.model.type](
        dim_projection = dim_projection[0],
        **block_kwargs
    )

    # Create EMA model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = model.to(device)
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    if args.compile.enabled:
        apply_compile(args.model.network_config.segments, model, args.compile)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if accelerator.is_main_process:
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Fixed parameters: {fixed_params}")

    # create loss function
    scheduler = blockwise_flow_matching(
        prediction=args.loss.prediction,
        path_type=args.loss.path_type,
        encoders=encoders,
        segments=args.model.network_config.segments,
        accelerator=accelerator,
    )

    # Setup optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.optimization.learning_rate,
        betas=(args.optimization.adam_beta1, args.optimization.adam_beta2),
        weight_decay=args.optimization.adam_weight_decay,
        eps=args.optimization.adam_epsilon,
    )
    if accelerator.is_main_process:
        logger.info(f"Optimizer initialized with learning rate: {args.optimization.learning_rate}")

    # Setup Dataloader
    train_dataset = CustomDataset(args.dataset.data_dir)
    local_batch_size = int(args.dataset.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.dataset.prefetch_factor,
        persistent_workers=args.dataset.persistent_workers
    )

    model = accelerator.prepare_model(model, device_placement=False)
    ema = accelerator.prepare_model(ema, device_placement=False)
    train_dataloader, optimizer = accelerator.prepare(train_dataloader, optimizer)

    global_step = 0
    if args.optimization.resume_step > 0:
        # Load checkpoint if resuming training
        dirs = os.listdir(checkpoint_dir)
        dirs = [d for d in dirs if d.isdigit()]
        dirs = sorted(dirs, key=lambda x: int(x))
        resume_from_path = dirs[-1] if len(dirs) > 0 else None

        if resume_from_path is None:
            if accelerator.is_main_process:
                logger.warning(f"No checkpoint found in {checkpoint_dir}. Starting from scratch.")
            global_step = 0
        else:
            if accelerator.is_main_process:
                logger.info(f"Resuming from checkpoint: {resume_from_path}")
            accelerator.load_state(os.path.join(checkpoint_dir, resume_from_path))
            global_step = int(resume_from_path)  # Use the last checkpoint as the global step

        update_ema_with_model = getattr(args.optimization, 'update_ema_with_model', False)
        if update_ema_with_model:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # create vae
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema", torch_dtype=weight_dtype).to(device)
    requires_grad(ema, False)
    if accelerator.is_main_process:
        logger.info(f"VAE loaded")

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)


    if accelerator.is_main_process and args.evaluation.eval_fid:
        import tensorflow.compat.v1 as tf
        hf_config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        hf_config.gpu_options.allow_growth = True
        hf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
        evaluator = Evaluator(tf.Session(config=hf_config), batch_size=20)
        ref_acts = evaluator.read_activations_npz(args.evaluation.ref_path)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.evaluation.ref_path, ref_acts)

    torch.cuda.empty_cache()
    if accelerator.is_main_process and args.evaluation.eval_fid:
        tf.reset_default_graph()

    # Train!
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {local_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.dataset.batch_size}")
        logger.info(f"  Learning rate = {args.optimization.learning_rate}")
        logger.info(f"  Gradient Accumulation steps = {args.optimization.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.optimization.max_train_steps}")
        logger.info(f"  Current optimization steps = {global_step}")
        logger.info(f"  Training Mixed-Precision = {args.optimization.mixed_precision}")

    # create progress bar
    progress_bar = tqdm(
        range(0, args.optimization.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 16
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    n = ys.size(0)
    xT = torch.randn((n, args.model.network_config.in_channels, latent_size, latent_size), device=device)
    segment_index = 0
    test_non_ema = args.evaluation.test_non_ema
    for epoch in range(args.optimization.epochs):
        for batch in train_dataloader:
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 3:
                raw_image, x, y = batch
            else:
                raise ValueError(f"Invalid batch length: {len(batch)}")
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            labels = y

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, args.loss.enc_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:]
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            with accelerator.accumulate(model):
                segment_idx = segment_index % args.model.network_config.segments
                model_kwargs = dict(y=labels)
                if args.model.network_config.finetune:
                    denoise_loss, proj_loss = scheduler.compute_loss_ft(model, x, model_kwargs, zs=zs[0], segment_idx=segment_idx)
                else:
                    denoise_loss, proj_loss = scheduler.compute_loss(model, x, model_kwargs, zs=zs[0], segment_idx=segment_idx)

                loss = denoise_loss + args.loss.proj_loss_weight * proj_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.optimization.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                segment_index += 1

            # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
            if accelerator.sync_gradients:
                update_ema(ema, model)
                progress_bar.update(1)
                global_step += 1

                if (global_step % args.evaluation.checkpointing_steps == 0 and global_step > 0):
                    checkpoints = os.listdir(checkpoint_dir)
                    checkpoints = [c for c in checkpoints if c.isdigit()]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x))
                    if accelerator.is_main_process and len(checkpoints) >= args.evaluation.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.evaluation.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[:num_to_remove]
                        logger.info(f"Removing old checkpoints")
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = f"{checkpoint_dir}/{removing_checkpoint}"
                            shutil.rmtree(removing_checkpoint)

                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}"

                    # Use clean checkpoint saving for both FSDP and non-FSDP models
                    if accelerator.is_main_process:
                        os.makedirs(checkpoint_path, exist_ok=True)
                    accelerator.wait_for_everyone()
                    # Save the model state
                    accelerator.save_state(checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    accelerator.wait_for_everyone()

                    if global_step in args.evaluation.checkpointing_steps_list:
                        checkpoint_path = os.path.join(checkpoint_dir, f"save-{global_step:07d}")
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            os.makedirs(checkpoint_path, exist_ok=True)
                        # Save the model state
                        accelerator.save_state(checkpoint_path)
                        if accelerator.is_main_process:
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                        accelerator.wait_for_everyone()

                if (global_step == 1 or (global_step % args.evaluation.sampling_steps == 0 and global_step > 0)):
                    model.eval()
                    with accelerator.autocast():
                        samples = scheduler.sample_ode(
                            model if test_non_ema else ema,
                            xT,
                            ys,
                            num_steps_per_segment=args.evaluation.evaluation_num_steps_per_segment,
                            cfg_scale=1.0,
                            guidance_low=0.,
                            guidance_high=1.,
                        ).to(weight_dtype)
                    samples = vae.decode(samples / vae.config.scaling_factor).sample
                    samples = samples.clamp(-1, 1)
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    torchvision.utils.save_image(out_samples, f'{gen_results_dir}/{global_step}.jpg', normalize=True, scale_each=True)
                    if accelerator.is_main_process:
                        logger.info("Generating EMA samples done.")

                    # CFG samples
                    torch.cuda.empty_cache()
                    with accelerator.autocast():
                        samples = scheduler.sample_ode(
                            model if test_non_ema else ema,
                            xT,
                            ys,
                            num_steps_per_segment=args.evaluation.evaluation_num_steps_per_segment,
                            cfg_scale=4.0,
                            guidance_low=0.,
                            guidance_high=1.,
                        ).to(weight_dtype)
                    samples = vae.decode(samples / vae.config.scaling_factor).sample
                    samples = samples.clamp(-1, 1)
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    torchvision.utils.save_image(out_samples, f'{gen_results_dir}/{global_step}_cfg.jpg', normalize=True, scale_each=True)
                    if accelerator.is_main_process:
                        logger.info("Generating EMA samples done.")
                    model.train()

                if args.evaluation.eval_fid and (global_step % args.evaluation.evaluation_steps == 0 and global_step > 0):
                    all_images = []
                    num_samples_per_process = int(args.evaluation.evaluation_number_samples // accelerator.num_processes)

                    number = 0
                    arr_list = []
                    test_fid_batch_size = args.evaluation.evaluation_batch_size
                    while num_samples_per_process > number:
                        latents = torch.randn((test_fid_batch_size, args.model.network_config.in_channels, latent_size, latent_size)).to(device=device)
                        y = torch.randint(0, 1000, (test_fid_batch_size,), device=device)
                        with torch.no_grad():
                            with accelerator.autocast():
                                samples = scheduler.sample_ode(
                                    ema,
                                    latents,
                                    y,
                                    num_steps_per_segment=args.evaluation.evaluation_num_steps_per_segment,
                                    cfg_scale=1.5,
                                    guidance_low=0.,
                                    guidance_high=1.,
                                ).to(weight_dtype)
                                samples = vae.decode(samples / vae.config.scaling_factor).sample
                                samples = samples.clamp(-1, 1)
                                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(torch.uint8).contiguous()
                                number += samples.shape[0]

                        gathered = accelerator.gather(samples)
                        all_images.append(gathered.cpu().numpy())

                    arr_list = np.concatenate(all_images, axis=0)
                    arr_list = arr_list[: int(args.evaluation.evaluation_number_samples)]

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info(f"Total number of samples gathered: {len(arr_list)}")
                        sample_acts, sample_stats, sample_stats_spatial = calculate_inception_stats_imagenet(arr_list, evaluator)
                        inception_score = evaluator.compute_inception_score(sample_acts[0])
                        fid = sample_stats.frechet_distance(ref_stats)
                        sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
                        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
                        logger.info("FID and Inception Score calculated.")
                        logger.info(f"Inception Score: {inception_score}")
                        logger.info(f"FID: {fid}")
                        logger.info(f"Spatial FID: {sfid}")
                        logger.info(f"Precision: {prec}")
                        logger.info(f"Recall: {recall}")
                        if args.logging.report_to == 'tensorboard' or args.logging.report_to == 'wandb':
                            accelerator.log({"inception_score": inception_score}, step=global_step)
                            accelerator.log({"fid": fid}, step=global_step)
                            accelerator.log({"sfid": sfid}, step=global_step)
                            accelerator.log({"prec": prec}, step=global_step)
                            accelerator.log({"recall": recall}, step=global_step)

            current_lr = optimizer.param_groups[0]['lr']
            logs = {"step_loss": denoise_loss.detach().item(),
                    "proj_loss": proj_loss.detach().item(),
                    "lr": current_lr}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.optimization.max_train_steps:
                break
        if global_step >= args.optimization.max_train_steps:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    return config

if __name__ == "__main__":
    args = parse_args()
    main(args)