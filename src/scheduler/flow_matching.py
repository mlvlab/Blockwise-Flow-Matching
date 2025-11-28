import torch
import numpy as np
import torch.nn.functional as F
import torch.cuda.amp as amp

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score

def compute_diffusion(t_cur):
    return 2 * t_cur

class blockwise_flow_matching:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            segments=6,
            accelerator=None,
            **kwargs,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.segments = segments
        self.sigmas = torch.linspace(1, 0, segments+1)
        self.accelerator = accelerator

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_loss(
        self,
        model,
        images,
        model_kwargs=None,
        zs=None,
        segment_idx=None,
    ):
        if model_kwargs == None:
            model_kwargs = {}

        denoising_loss = 0.0
        proj_loss = 0.0

        batch_size = images.shape[0]
        noises = torch.randn_like(images)

        # Split batch across segments as evenly as possible
        num_segments = self.segments
        base = batch_size // num_segments
        remainder = batch_size % num_segments
        split_sizes = [base + 1 if i < remainder else base for i in range(num_segments)]

        time_input_list = []
        for seg_idx, split_size in enumerate(split_sizes):
            if split_size == 0:
                continue
            sigma_next = self.sigmas[seg_idx + 1]
            sigma_current = self.sigmas[seg_idx]
            time_input_list.append(torch.empty((split_size,)).uniform_(sigma_next.item(), sigma_current.item()).to(images.device))

        time_input = torch.cat(time_input_list, dim=0)
        while len(time_input.shape) < images.ndim:
            time_input = time_input.unsqueeze(-1)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        #model_kwargs['segment_idx'] = segment_idx
        with self.accelerator.autocast():
            model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs, split_sizes=split_sizes)

        denoising_loss_per_iter = mean_flat((model_output - model_target) ** 2)
        denoising_loss += denoising_loss_per_iter.mean()

        # projection loss
        if self.encoders is not None and zs_tilde is not None:
            proj_loss_per = 0.0
            for j, (repre_j, raw_j) in enumerate(zip(zs_tilde, zs)):
                raw_j_norm = F.normalize(raw_j, dim=-1)
                repre_j_norm = F.normalize(repre_j, dim=-1)
                proj_loss_per += mean_flat(-(raw_j_norm * repre_j_norm).sum(dim=-1))
            proj_loss += proj_loss_per / zs.shape[0]
            # # Vectorized normalization
            # z_norm = F.normalize(zs, dim=-1)
            # z_tilde_norm = F.normalize(zs_tilde, dim=-1)

            # # Vectorized loss calculation
            # # (B, N, D) * (B, N, D) -> sum over D -> (B, N)
            # loss_per_token = -(z_norm * z_tilde_norm).sum(dim=-1)

            # # Mean over N and B dimensions
            # proj_loss_per_iter = loss_per_token.mean()
            # proj_loss += proj_loss_per_iter

        return denoising_loss, proj_loss

    def compute_loss_ft(
        self,
        model,
        images,
        model_kwargs=None,
        zs=None,
        segment_idx=None,
    ):
        if model_kwargs == None:
            model_kwargs = {}

        denoising_loss = 0.0
        proj_loss = 0.0

        batch_size = images.shape[0]
        noises = torch.randn_like(images)

        # Split batch across segments as evenly as possible
        num_segments = self.segments
        base = batch_size // num_segments
        remainder = batch_size % num_segments
        split_sizes = [base + 1 if i < remainder else base for i in range(num_segments)]

        time_input_list = []
        time_start_input_list = []
        time_end_input_list = []
        for seg_idx, split_size in enumerate(split_sizes):
            if split_size == 0:
                continue
            sigma_next = self.sigmas[seg_idx + 1]
            sigma_current = self.sigmas[seg_idx]
            time_start_input_list.append(torch.ones((split_size,), device=images.device) * sigma_current)
            time_end_input_list.append(torch.ones((split_size,), device=images.device) * sigma_next)
            time_input_list.append(torch.empty((split_size,)).uniform_(sigma_next.item(), sigma_current.item()).to(images.device))

        time_input = torch.cat(time_input_list, dim=0)
        while len(time_input.shape) < images.ndim:
            time_input = time_input.unsqueeze(-1)
        time_start_input = torch.cat(time_start_input_list, dim=0)
        time_end_input = torch.cat(time_end_input_list, dim=0)
        while len(time_start_input.shape) < images.ndim:
            time_start_input = time_start_input.unsqueeze(-1)
        while len(time_end_input.shape) < images.ndim:
            time_end_input = time_end_input.unsqueeze(-1)

        coeff = (time_start_input - time_input) / (time_start_input - time_end_input)
        coeff = coeff.squeeze(1)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        alpha_start_t, sigma_start_t, d_alpha_start_t, d_sigma_start_t = self.interpolant(time_start_input)

        model_input = alpha_t * images + sigma_t * noises
        model_input_start = alpha_start_t * images + sigma_start_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        model_kwargs['t_start'] = time_start_input.flatten()
        model_kwargs['x_start'] = model_input_start
        model_kwargs['coeff'] = coeff

        with self.accelerator.autocast():
            approx, representation_target = model(model_input, time_input.flatten(), **model_kwargs, split_sizes=split_sizes)

        denoising_loss_per_iter = mean_flat((approx - representation_target) ** 2)
        denoising_loss += denoising_loss_per_iter.mean()

        return denoising_loss, proj_loss

    @torch.no_grad()
    def compute_loss_analysis(
        self,
        model,
        images,
        model_kwargs=None,
        zs=None,
        segment_idx=None,
    ):
        if model_kwargs == None:
            model_kwargs = {}

        denoising_loss = 0.0
        proj_loss = 0.0

        batch_size = images.shape[0]
        noises = torch.randn_like(images)

        # Split batch across segments as evenly as possible
        num_segments = self.segments
        base = batch_size // num_segments
        remainder = batch_size % num_segments
        split_sizes = [base + 1 if i < remainder else base for i in range(num_segments)]

        time_input_list = []
        for seg_idx, split_size in enumerate(split_sizes):
            if split_size == 0:
                continue
            sigma_next = self.sigmas[seg_idx + 1]
            sigma_current = self.sigmas[seg_idx]
            #time_input_list.append(torch.empty((split_size,)).uniform_(sigma_next.item(), sigma_current.item()).to(images.device))
            time_input_list.append(torch.linspace(sigma_next.item(), sigma_current.item(), split_size, device=images.device))


        time_input = torch.cat(time_input_list, dim=0)
        while len(time_input.shape) < images.ndim:
            time_input = time_input.unsqueeze(-1)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        #model_kwargs['segment_idx'] = segment_idx
        with self.accelerator.autocast():
            model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs, split_sizes=split_sizes)

        denoising_loss_per_iter = mean_flat((model_output - model_target) ** 2)
        segment_wise_loss = torch.split(denoising_loss_per_iter, split_sizes, dim=0)

        return segment_wise_loss, None

    def sample_ode(
        self,
        model,
        latents,
        y,
        num_steps_per_segment=10,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
    ):
        # setup conditioning
        if cfg_scale > 1.0:
            y_null = torch.tensor([1000] * y.size(0), device=y.device)
        _dtype = latents.dtype
        x_next = latents.to(torch.float64)
        device = x_next.device

        with torch.no_grad():
            for segment_idx in range(self.segments):
                sigma_next = self.sigmas[segment_idx + 1]
                sigma_current = self.sigmas[segment_idx]
                sigma_list = torch.linspace(sigma_current, sigma_next, num_steps_per_segment+1)

                for j, (t_cur, t_next) in enumerate(zip(sigma_list[:-1], sigma_list[1:])):
                    x_cur = x_next
                    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                        model_input = torch.cat([x_cur] * 2, dim=0)
                        y_cur = torch.cat([y, y_null], dim=0)
                    else:
                        model_input = x_cur
                        y_cur = y
                    kwargs = dict(y=y_cur, segment_idx=segment_idx)
                    time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
                    d_cur = model(
                        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                        )[0].to(torch.float64)

                    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
                    x_next = x_cur + (t_next - t_cur) * d_cur
        return x_next

    def sample_ode_frn(
        self,
        model,
        latents,
        y,
        num_steps_per_segment=10,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
    ):
        # setup conditioning
        if cfg_scale > 1.0:
            y_null = torch.tensor([1000] * y.size(0), device=y.device)
        _dtype = latents.dtype
        x_next = latents.to(torch.float64)
        device = x_next.device

        with torch.no_grad():
            for segment_idx in range(self.segments):
                sigma_next = self.sigmas[segment_idx + 1]
                sigma_current = self.sigmas[segment_idx]
                sigma_list = torch.linspace(sigma_current, sigma_next, num_steps_per_segment+1)

                for j, (t_cur, t_next) in enumerate(zip(sigma_list[:-1], sigma_list[1:])):
                    x_cur = x_next
                    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                        model_input = torch.cat([x_cur] * 2, dim=0)
                        y_cur = torch.cat([y, y_null], dim=0)
                    else:
                        model_input = x_cur
                        y_cur = y
                    if j == 0:
                        representation_feature = None
                    time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
                    time_start_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * sigma_current
                    time_end_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * sigma_next
                    coeff = (time_start_input - time_input) / (time_start_input - time_end_input)
                    coeff = coeff.reshape(coeff.size(0), 1, 1)
                    kwargs = dict(y=y_cur, segment_idx=segment_idx, representation_feature=representation_feature, coeff=coeff)
                    d_cur, representation_feature = model(
                        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                        )

                    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
                    x_next = x_cur + (t_next - t_cur) * d_cur
        return x_next

    def sample_sde(
        self,
        model,
        latents,
        y,
        num_steps_per_segment=10,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
    ):
        if cfg_scale > 1.0:
            y_null = torch.tensor([1000] * y.size(0), device=y.device)

        _dtype = latents.dtype
        sigmas = torch.linspace(1., 0.04, self.segments+1, dtype=torch.float64)
        sigmas = torch.cat([sigmas, torch.tensor([0.], dtype=torch.float64)])
        x_next = latents.to(torch.float64)
        device = x_next.device

        with torch.no_grad():
            for segment_idx in range(self.segments):
                if segment_idx == self.segments - 1:
                    sigma_current = self.sigmas[segment_idx]
                    sigma_next = 0.04
                    sigma_list = torch.linspace(sigma_current, sigma_next, num_steps_per_segment)
                    sigma_list = torch.cat([sigma_list, torch.tensor([0.])])
                else:
                    sigma_next = self.sigmas[segment_idx + 1]
                    sigma_current = self.sigmas[segment_idx]
                    sigma_list = torch.linspace(sigma_current, sigma_next, num_steps_per_segment+1)

                for j, (t_cur, t_next) in enumerate(zip(sigma_list[:-1], sigma_list[1:])):
                    dt = t_next - t_cur
                    x_cur = x_next
                    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                        model_input = torch.cat([x_cur] * 2, dim=0)
                        y_cur = torch.cat([y, y_null], dim=0)
                    else:
                        model_input = x_cur
                        y_cur = y
                    kwargs = dict(y=y_cur, segment_idx=segment_idx)
                    time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
                    diffusion = compute_diffusion(t_cur)
                    eps_i = torch.randn_like(x_cur).to(device)
                    deps = eps_i * torch.sqrt(torch.abs(dt))

                    # compute drift
                    v_cur = model(
                        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                        )[0].to(torch.float64)
                    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=self.path_type)
                    d_cur = v_cur - 0.5 * diffusion * s_cur
                    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

                    x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

            t_cur, t_next = sigma_list[-2], sigma_list[-1]
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(y=y_cur, segment_idx=segment_idx)
            time_input = torch.ones(model_input.size(0)).to(
                device=device, dtype=torch.float64
                ) * t_cur

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=self.path_type)
            diffusion = compute_diffusion(t_cur)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            mean_x = x_cur + dt * d_cur

            return mean_x