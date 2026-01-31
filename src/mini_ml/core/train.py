from datetime import datetime
import logging
import random
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
import yaml
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader

from mini_ml.utils.config import AppConfig

from mini_ml.core.factory import (
    build_dataset,
    build_loss,
    build_lr_scheduler,
    build_model,
    build_optimizer,
)

logger = logging.getLogger(__name__)


def setup_seed(seed: Optional[int]):
    """设置随机种子以保证实验在相同条件下可复现。"""
    if seed is None:
        seed = int(datetime.now().timestamp())  # 或 random.randint(0, 2**32 - 1)
        print(f"[INFO] No seed provided, generated seed = {seed}")
    else:
        print(f"[INFO] Using fixed seed = {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 保证 cudnn 的确定性，这可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def train(config: AppConfig):
    """
    主训练函数，接收一个经过验证的 Pydantic 配置对象。
    """
    # ==========================================================
    # 1. 准备环境: 输出目录, 日志, 随机种子, 设备
    # ==========================================================
    output_dir = Path(config.environment.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(config.environment.seed)

    logger.info("All outputs will be saved to: %s", output_dir)

    # 将最终生效的配置也保存一份到输出目录，方便复现！
    with open(output_dir / "config_final.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, indent=2, sort_keys=False)
    logger.info("Final configuration saved to config_final.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # TensorBoard
    # tensorboard_dir = output_dir / "tensorboard_logs"
    # tensorboard_dir.mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(log_dir=str(tensorboard_dir))
    # logger.info("TensorBoard logs will be saved to: %s", tensorboard_dir)
    writer = SummaryWriter()

    # ==========================================================
    # 2. 构建所有组件
    # ==========================================================
    logger.info("Building components...")
    model = build_model(config.model).to(device)
    criterion = build_loss(config.loss)
    optimizer = build_optimizer(config.optimizer, model)
    lr_scheduler = build_lr_scheduler(
        config.lr_scheduler, optimizer, max_iters=config.iters
    )
    train_dataset = build_dataset(config.dataset, mode="train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.environment.num_workers,
        pin_memory=True,  # 如果使用 GPU，建议开启
    )
    logger.info("Train dataset created with %d samples.", len(train_dataset))
    logger.info("Components built successfully.")

    # ==========================================================
    # 3. 训练循环
    # ==========================================================
    logger.info("Starting training...")
    max_iters = config.iters
    log_interval = config.environment.log_interval
    save_interval = config.environment.save_interval

    model.train()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("Iter: {task.completed} of {task.total}"),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=True,  # 训练结束后进度条会消失
    ) as progress:

        train_task = progress.add_task("[green]Training", total=max_iters)
        current_iter = 0
        data_iterator = iter(train_loader)
        while current_iter < max_iters:
            try:
                images, masks = next(data_iterator)
            except StopIteration:
                # 如果 DataLoader 遍历完了一轮，就重新创建迭代器
                data_iterator = iter(train_loader)
                images, masks = next(data_iterator)

            images = images.to(device)
            masks = masks.to(device)

            # --- 核心训练 ---
            outputs = model(images)
            loss_dict = criterion(outputs["pred"], masks)
            total_loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            current_iter += 1
            progress.update(train_task, advance=1)

            if current_iter % log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                loss_log_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in loss_dict.items() if k != "total_loss"]
                )
                logger.info(
                    f"Iter: [{current_iter}/{max_iters}] | LR: {current_lr:.6f} | Loss: {total_loss.item():.4f} ({loss_log_str})"
                )

                # <<< TensorBoard 记录 >>>
                # 记录总损失
                writer.add_scalar("Loss/total", total_loss.item(), current_iter)
                # 记录学习率
                writer.add_scalar("Metrics/Learning_Rate", current_lr, current_iter)
                # 记录各个分项损失
                for k, v in loss_dict.items():
                    if k != "total_loss":
                        writer.add_scalar(f"Loss/{k}", v, current_iter)

            # --- 定期保存模型检查点 ---
            if current_iter > 0 and current_iter % save_interval == 0:
                ckpt_path = checkpoints_dir / f"checkpoint_iter_{current_iter}.pth"
                torch.save(model.state_dict(), ckpt_path)
                logger.info("Checkpoint saved to {ckpt_path}")

    # ==========================================================
    # 4. 保存最终结果
    # ==========================================================
    logger.info("Training Finished.")
    final_model_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info("Final model saved to: %s", final_model_path)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the training configuration YAML file.",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir_override",
    type=click.Path(resolve_path=True),
    default=None,
    help="Override the output directory specified in the config file.",
)
def main(config_path: str, output_dir_override: str):
    """
    一个基于配置文件的、可复现的深度学习模型训练脚本。

    示例:

    python train.py -c configs/my_experiment.yaml

    python train.py -c configs/my_experiment.yaml -o outputs/new_run
    """
    # 1. 加载 YAML 配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # 2. 如果命令行提供了输出目录，就用它覆盖配置文件中的值
    if output_dir_override:
        # 确保 environment 键存在
        if "environment" not in raw_config:
            raw_config["environment"] = {}
        raw_config["environment"]["output_dir"] = output_dir_override

    # 3. 使用 Pydantic 解析和验证最终的配置
    try:
        config = AppConfig(**raw_config)
    except Exception:
        # 使用 rich 打印出漂亮的错误回溯，帮助快速定位配置问题
        console = Console()
        console.print("❌ [bold red]Error: Configuration validation failed![/bold red]")
        console.print(
            "Please check your YAML file against the Pydantic models in config.py."
        )
        console.print_exception(show_locals=True)
        return

    # 4. 调用主训练函数，传入经过验证和覆盖的配置对象
    train(config)


if __name__ == "__main__":
    main()
