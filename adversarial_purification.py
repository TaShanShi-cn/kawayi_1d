import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple

from models.unet_1d import Model1D
from rml2018_dataset import RML2018Dataset
from diffpure_ddpm_1d import Diffusion1D


class SignalClassifier(nn.Module):
    """用于RML2018信号分类的简单CNN分类器"""

    def __init__(self, num_classes=11, input_channels=2, signal_length=1024):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        # 计算全连接层输入维度
        self.flatten_dim = 256 * (signal_length // 32)

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class PGDAttack:
    """PGD对抗攻击"""

    def __init__(self, model, eps=0.1, alpha=0.01, steps=10, device='cpu'):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device

    def attack(self, x, y):
        """生成对抗样本"""
        x_adv = x.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x_adv, -1, 1)

        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            with torch.enable_grad():
                outputs = self.model(x_adv)
                loss = F.cross_entropy(outputs, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, min=-1, max=1)

        return x_adv


def train_classifier(config, args):
    """训练分类器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training classifier on device: {device}")

    # 创建数据加载器
    train_dataset = RML2018Dataset(args.data_path, split='train', normalize=True)
    val_dataset = RML2018Dataset(args.data_path, split='val', normalize=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    # 创建分类器
    # 假设RML2018有11个调制类型
    classifier = SignalClassifier(num_classes=11).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(50):  # 训练50个epoch
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Training Classifier Epoch {epoch + 1}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({
                'Loss': f'{train_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        # 验证
        if epoch % 10 == 9:
            classifier.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = classifier(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            print(f'Validation Acc: {100. * correct / total:.2f}%')

    # 保存分类器
    torch.save(classifier.state_dict(), os.path.join(args.save_dir, 'classifier.pth'))
    return classifier


def evaluate_purification(config, args):
    """评估对抗样本净化效果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating purification on device: {device}")

    # 加载分类器
    classifier = SignalClassifier(num_classes=11).to(device)
    if os.path.exists(os.path.join(args.save_dir, 'classifier.pth')):
        classifier.load_state_dict(torch.load(os.path.join(args.save_dir, 'classifier.pth')))
        print("Loaded pretrained classifier")
    else:
        print("Training classifier first...")
        classifier = train_classifier(config, args)

    classifier.eval()

    # 初始化DDPM净化器
    ddpm_args = argparse.Namespace(
        model_path=args.ddmp_model_path,
        log_dir=args.log_dir,
        sample_step=1,  # 净化步骤
        t=args.purification_steps  # 扩散步数
    )

    diffusion = Diffusion1D(ddmp_args, config, device)

    # 创建测试数据加载器
    test_dataset = RML2018Dataset(args.data_path, split='test', normalize=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # 创建PGD攻击器
    attacker = PGDAttack(classifier, eps=args.attack_eps, steps=args.attack_steps, device=device)

    # 评估指标
    clean_correct = 0
    adv_correct = 0
    purified_correct = 0
    total = 0

    # 存储结果用于可视化
    results = {
        'clean_signals': [],
        'adv_signals': [],
        'purified_signals': [],
        'labels': []
    }

    print("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # 1. 干净样本分类
            clean_output = classifier(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()

            # 2. 生成对抗样本
            adv_data = attacker.attack(data, target)
            adv_output = classifier(adv_data)
            adv_pred = adv_output.argmax(dim=1)
            adv_correct += adv_pred.eq(target).sum().item()

            # 3. DDPM净化
            purified_data = diffusion.signal_editing_sample(
                signal=adv_data,
                bs_id=batch_idx,
                tag=f'test_batch_{batch_idx}'
            )

            # 只取第一个净化结果（如果有多个sample_step）
            if purified_data.size(0) > batch_size:
                purified_data = purified_data[:batch_size]

            purified_output = classifier(purified_data)
            purified_pred = purified_output.argmax(dim=1)
            purified_correct += purified_pred.eq(target).sum().item()

            total += batch_size

            # 存储前几个batch的结果用于可视化
            if batch_idx < 3:
                results['clean_signals'].append(data.cpu())
                results['adv_signals'].append(adv_data.cpu())
                results['purified_signals'].append(purified_data.cpu())
                results['labels'].append(target.cpu())

            # 打印进度
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Clean Acc: {100. * clean_correct / total:.2f}%, "
                      f"Adv Acc: {100. * adv_correct / total:.2f}%, "
                      f"Purified Acc: {100. * purified_correct / total:.2f}%")

    # 最终结果
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    purified_acc = 100. * purified_correct / total

    print(f"\nFinal Results:")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Purified Accuracy: {purified_acc:.2f}%")
    print(f"Purification Recovery: {purified_acc - adv_acc:.2f}%")

    # 可视化结果
    visualize_results(results, args.save_dir)

    return clean_acc, adv_acc, purified_acc


def visualize_results(results, save_dir):
    """可视化净化结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 拼接所有batch的数据
    clean_signals = torch.cat(results['clean_signals'], dim=0)
    adv_signals = torch.cat(results['adv_signals'], dim=0)
    purified_signals = torch.cat(results['purified_signals'], dim=0)

    # 选择前5个样本进行可视化
    n_samples = min(5, clean_signals.size(0))

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # I通道
        axes[i, 0].plot(clean_signals[i, 0].numpy(), 'b-', alpha=0.7, label='Clean I')
        axes[i, 0].plot(clean_signals[i, 1].numpy(), 'r-', alpha=0.7, label='Clean Q')
        axes[i, 0].set_title(f'Sample {i + 1}: Clean Signal')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(adv_signals[i, 0].numpy(), 'b-', alpha=0.7, label='Adv I')
        axes[i, 1].plot(adv_signals[i, 1].numpy(), 'r-', alpha=0.7, label='Adv Q')
        axes[i, 1].set_title(f'Sample {i + 1}: Adversarial Signal')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        axes[i, 2].plot(purified_signals[i, 0].numpy(), 'b-', alpha=0.7, label='Purified I')
        axes[i, 2].plot(purified_signals[i, 1].numpy(), 'r-', alpha=0.7, label='Purified Q')
        axes[i, 2].set_title(f'Sample {i + 1}: Purified Signal')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'purification_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 计算并可视化信号差异
    adv_noise = adv_signals - clean_signals
    purified_diff = purified_signals - clean_signals

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 对抗噪声分布
    axes[0, 0].hist(adv_noise.numpy().flatten(), bins=50, alpha=0.7, color='red')
    axes[0, 0].set_title('Adversarial Noise Distribution')
    axes[0, 0].set_xlabel('Noise Amplitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 净化后差异分布
    axes[0, 1].hist(purified_diff.numpy().flatten(), bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Purified Signal Difference Distribution')
    axes[0, 1].set_xlabel('Difference Amplitude')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # MSE对比
    mse_adv = torch.mean((adv_signals - clean_signals) ** 2, dim=[1, 2])
    mse_purified = torch.mean((purified_signals - clean_signals) ** 2, dim=[1, 2])

    axes[1, 0].scatter(range(len(mse_adv)), mse_adv.numpy(), alpha=0.6, color='red', label='Adversarial MSE')
    axes[1, 0].scatter(range(len(mse_purified)), mse_purified.numpy(), alpha=0.6, color='green', label='Purified MSE')
    axes[1, 0].set_title('MSE Comparison')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 频谱分析（显示第一个样本的FFT）
    if len(clean_signals) > 0:
        clean_fft = torch.fft.fft(clean_signals[0, 0] + 1j * clean_signals[0, 1])
        adv_fft = torch.fft.fft(adv_signals[0, 0] + 1j * adv_signals[0, 1])
        purified_fft = torch.fft.fft(purified_signals[0, 0] + 1j * purified_signals[0, 1])

        freqs = torch.fft.fftfreq(clean_signals.size(-1))

        axes[1, 1].plot(freqs.numpy(), torch.abs(clean_fft).numpy(), 'b-', alpha=0.7, label='Clean')
        axes[1, 1].plot(freqs.numpy(), torch.abs(adv_fft).numpy(), 'r-', alpha=0.7, label='Adversarial')
        axes[1, 1].plot(freqs.numpy(), torch.abs(purified_fft).numpy(), 'g-', alpha=0.7, label='Purified')
        axes[1, 1].set_title('Frequency Domain Comparison')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'signal_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Adversarial Purification for RML2018 Signals')
    parser.add_argument('--config', type=str, required=True, help='Path to DDPM config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to RML2018 dataset')
    parser.add_argument('--ddmp_model_path', type=str, required=True, help='Path to trained DDPM model')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logs')

    # 攻击参数
    parser.add_argument('--attack_eps', type=float, default=0.1, help='PGD attack epsilon')
    parser.add_argument('--attack_steps', type=int, default=10, help='PGD attack steps')

    # 净化参数
    parser.add_argument('--purification_steps', type=int, default=100, help='DDPM purification steps')

    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train_classifier', 'evaluate', 'both'],
                        default='both', help='Mode to run')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 转换为命名元组
    def dict_to_namedtuple(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = dict_to_namedtuple(value)
        return namedtuple('Config', d.keys())(**d)

    config = dict_to_namedtuple(config)

    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.mode in ['train_classifier', 'both']:
        print("Training classifier...")
        train_classifier(config, args)

    if args.mode in ['evaluate', 'both']:
        print("Evaluating purification...")
        clean_acc, adv_acc, purified_acc = evaluate_purification(config, args)

        # 保存结果
        results = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'purified_accuracy': purified_acc,
            'purification_recovery': purified_acc - adv_acc,
            'attack_eps': args.attack_eps,
            'attack_steps': args.attack_steps,
            'purification_steps': args.purification_steps
        }

        import json
        with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {args.save_dir}/results.json")


if __name__ == '__main__':
    main()
