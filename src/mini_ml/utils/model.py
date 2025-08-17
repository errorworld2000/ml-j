import logging
import os
import requests
from torch import nn
import safetensors.torch
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)

MODELS_DIR = "models"


def download_model_file(pretrained_model_url: str) -> None:
    """
    下载模型文件。

    Args:
        pretrained_model_url (str): 模型文件的 URL。
    """
    output_file = os.path.join(MODELS_DIR, pretrained_model_url.split("/")[-1])

    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(output_file):
        logger.info("文件已存在，跳过下载: %s", output_file)
        return

    logger.info("正在从 %s 下载模型文件...", pretrained_model_url)
    try:
        with requests.get(pretrained_model_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with (
                open(output_file, "wb") as f,
                Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TimeRemainingColumn(),
                ) as progress,
            ):
                task = progress.add_task("下载中...", total=total)
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        logger.info("模型文件已下载到: %s", output_file)
    except requests.RequestException as e:
        logger.error("模型文件下载失败: %s", e)
    return


def load_pretrained_model(model: nn.Module, pretrained_file: str) -> None:
    """
    加载预训练模型权重到模型实例。

    Args:
        model (torch.nn.Module): 要加载权重的模型。
        pretrained_file (str): 已下载的预训练模型文件路径。
    """
    if not os.path.exists(pretrained_file):
        logger.error("预训练文件不存在: %s", pretrained_file)
        return

    logger.info("从文件 %s 加载预训练模型...", pretrained_file)
    param_state_dict = safetensors.torch.load_file(pretrained_file)
    model_state_dict = model.state_dict()

    num_params_loaded = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("加载权重中...", total=len(model_state_dict))

        for k, param in model_state_dict.items():
            if k not in param_state_dict:
                logger.warning("%s 在预训练模型中未找到，跳过", k)
                progress.update(task, advance=1)
                continue

            if param.shape != param_state_dict[k].shape:
                logger.warning(
                    "[跳过] 参数 %s 的形状不匹配 (预训练: %s, 实际: %s)",
                    k,
                    param_state_dict[k].shape,
                    param.shape,
                )
                progress.update(task, advance=1)
                continue

            model_state_dict[k] = param_state_dict[k]
            num_params_loaded += 1
            progress.update(task, advance=1)

    model.load_state_dict(model_state_dict)
    logger.info(
        "成功加载了 %d/%d 个参数到模型 %s。",
        num_params_loaded,
        len(model_state_dict),
        model.__class__.__name__,
    )
    return
