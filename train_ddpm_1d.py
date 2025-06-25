import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from collections import namedtuple

from models.unet_1d import Model1D
from rml2018_dataset import RML2018Dataset
from diffpure_ddpm_1d import get_beta_schedule


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


class DDPMTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # 初始化模型
        self.model = Model1D(config).to(device)

        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate
        )

        # 初始化调度器
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step / config.training.warmup, 1.0)
        )

        # 扩散参数
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )

        self.betas = torch.from_numpy(betas).float().to(device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # EMA模型
        if config.model.ema:
            self.ema_model = Model1D(config).to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_rate = config.model.ema_rate
        else:
            self.ema_model = None

    def update_ema(self):
        """更新EMA模型"""
        if self.ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data.mul_(self.ema_rate).add_(param.data, alpha=1 - self.ema_rate)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def train_step(self, batch):
        """单步训练"""
        x_start, _ = batch  # 忽略标签，我们只需要信号数据
        x_start = x_start.to(self.device)

        batch_size = x_start.shape[0]

        # 随机采样时间步
        t = torch.randint(0, self.config.diffusion.num_diffusion_timesteps,
                          (batch_size,), device=self.device).long()

        # 添加噪声
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # 预测噪声
        predicted_noise = self.model(x_noisy, t)

        # 计算损失（MSE loss）
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if hasattr(self.config.training, 'grad_clip'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.config.training.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        # 更新EMA
        self.update_ema()

        return loss.item()

    def save_checkpoint(self, epoch, save_dir):
        """保存检查点"""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }

        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))

        # 只保存模型权重（用于推理）
        model_to_save = self.ema_model if self.ema_model is not None else self.model
        torch.save(model_to_save.state_dict(),
                   os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.ema_model is not None and 'ema_model_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        return checkpoint['epoch']


def train(config, args):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置日志
    setup_logging(args.log_dir)
    logging.info(f"Starting training with config: {config}")

    # 创建数据加载器
    train_dataset = RML2018Dataset(
        data_path=args.data_path,
        split='train',
        normalize=True
    )

    val_dataset = RML2018Dataset(
        data_path=args.data_path,
        split='val',
        normalize=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # 初始化训练器
    trainer = DDPMTrainer(config, device)

    # 如果有预训练模型，加载它
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.load_checkpoint(args.resume)
        logging.info(f"Resumed training from epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, config.training.n_epochs):
        # 训练阶段
        trainer.model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.training.n_epochs}')
        for batch in pbar:
            loss = trainer.train_step(batch)
            train_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.6f}'})

        avg_train_loss = np.mean(train_losses)

        # 验证阶段
        if len(val_loader) > 0:
            trainer.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    x_start, _ = batch
                    x_start = x_start.to(device)
                    batch_size = x_start.shape[0]

                    t = torch.randint(0, config.diffusion.num_diffusion_timesteps,
                                      (batch_size,), device=device).long()
                    noise = torch.randn_like(x_start)
                    x_noisy = trainer.q_sample(x_start, t, noise)
                    predicted_noise = trainer.model(x_noisy, t)
                    loss = nn.functional.mse_loss(predicted_noise, noise)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            logging.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        else:
            logging.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}")

        # 保存检查点
        if (epoch + 1) % config.training.save_freq == 0:
            trainer.save_checkpoint(epoch + 1, args.save_dir)
            logging.info(f"Saved checkpoint at epoch {epoch + 1}")

    # 保存最终模型
    trainer.save_checkpoint(config.training.n_epochs, args.save_dir)
    logging.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train DDPM for 1D signals')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to RML2018 dataset')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 转换为命名元组以便于访问
    def dict_to_namedtuple(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = dict_to_namedtuple(value)
        return namedtuple('Config', d.keys())(**d)

    config = dict_to_namedtuple(config)

    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 开始训练
    train(config, args)


if __name__ == '__main__':
    main()
