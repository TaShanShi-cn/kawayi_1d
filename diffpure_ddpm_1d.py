import os
import random
import numpy as np
import torch
from models.unet_1d import Model1D


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def signal_editing_denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    """
    Sample from p(x_{t-1} | x_t) for 1D signals
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class Diffusion1D(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        print("Loading 1D signal diffusion model")

        # 初始化1D UNet模型
        model = Model1D(self.config)

        # 如果有预训练权重，加载它们
        if hasattr(args, 'model_path') and args.model_path:
            if os.path.exists(args.model_path):
                ckpt = torch.load(args.model_path, map_location='cpu')
                model.load_state_dict(ckpt)
                print(f"Loaded pretrained model from {args.model_path}")
            else:
                print(f"Warning: Model path {args.model_path} does not exist, using random initialization")

        model.eval()
        self.model = model

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def signal_editing_sample(self, signal=None, bs_id=0, tag=None):
        assert isinstance(signal, torch.Tensor)
        batch_size = signal.shape[0]

        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert signal.ndim == 3, f"Expected 3D tensor (batch, channels, length), got {signal.ndim}D"
            x0 = signal

            # 保存目录
            if bs_id < 2:
                os.makedirs(out_dir, exist_ok=True)
                # 可以选择保存原始信号
                torch.save(x0, os.path.join(out_dir, f'original_input.pt'))

            xs = []
            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0).to(x0.device)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                if bs_id < 2:
                    torch.save(x, os.path.join(out_dir, f'init_{it}.pt'))

                # 反向扩散去噪过程
                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=signal.device)
                    x = signal_editing_denoising_step_flexible_mask(x, t=t, model=self.model,
                                                                    logvar=self.logvar,
                                                                    betas=self.betas.to(signal.device))
                    # 保存中间结果
                    if (i - 49) % 50 == 0 and bs_id < 2:
                        torch.save(x, os.path.join(out_dir, f'noise_t_{i}_{it}.pt'))

                x0 = x

                if bs_id < 2:
                    torch.save(x0, os.path.join(out_dir, f'samples_{it}.pt'))

                xs.append(x0)

            return torch.cat(xs, dim=0)
