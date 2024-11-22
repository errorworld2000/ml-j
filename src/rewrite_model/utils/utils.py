import contextlib
import os
import tempfile
from urllib import response
import requests
import safetensors.torch
import torch
from torch.hub import load_state_dict_from_url
from rewrite_model.utils.logger import logger
from safetensors import safe_open
from huggingface_hub import hf_hub_download


def download_model_file(pretrained_model_url, output_file=None):
    """
    下载模型文件。

    Args:
        pretrained_model_url (str): 模型文件的 URL。
        output_file (str, optional): 保存的文件名。如果未提供，则使用 URL 中的文件名。

    Returns:
        str: 下载的文件路径，如果下载失败则返回 None。
    """
    if output_file is None:
        output_file = pretrained_model_url.split("/")[-1]

    if os.path.exists(output_file):
        logger.info("文件已存在，跳过下载: %s", output_file)
        return output_file

    logger.info("正在从 %s 下载模型文件...", pretrained_model_url)
    response = requests.get(pretrained_model_url)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        logger.info("模型文件已下载到: %s", output_file)
        return output_file
    else:
        logger.error("模型文件下载失败，状态码: %d", response.status_code)
        return None


def load_pretrained_model(model, pretrained_file):
    """
    加载预训练模型权重到模型实例。

    Args:
        model (torch.nn.Module): 要加载权重的模型。
        pretrained_file (str): 已下载的预训练模型文件路径。

    Returns:
        int: 成功加载的参数个数。
    """
    if not os.path.exists(pretrained_file):
        logger.error("预训练文件不存在: %s", pretrained_file)
        return 0

    logger.info("从文件 %s 加载预训练模型...", pretrained_file)
    param_state_dict = safetensors.torch.load_file(pretrained_file)
    model_state_dict = model.state_dict()

    num_params_loaded = 0

    for k, param in model_state_dict.items():
        if k not in param_state_dict:
            logger.warning("%s 在预训练模型中未找到，跳过", k)
            continue

        if param.shape != param_state_dict[k].shape:
            logger.warning(
                "[跳过] 参数 %s 的形状不匹配 (预训练: %s, 实际: %s)",
                k,
                param_state_dict[k].shape,
                param.shape,
            )
            continue

        model_state_dict[k] = param_state_dict[k]
        num_params_loaded += 1

    model.load_state_dict(model_state_dict)

    logger.info(
        "成功加载了 %d/%d 个参数到模型 %s。",
        num_params_loaded,
        len(model_state_dict),
        model.__class__.__name__,
    )

    return num_params_loaded


@contextlib.contextmanager
def generate_tempdir(directory: str | None = None, **kwargs):
    """Generate a temporary directory"""
    directory = seg_env.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir
