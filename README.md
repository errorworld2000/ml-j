# mini-ml

一个用于模型重写的小型项目。

## ✨ 功能清单

- [ ] 训练(Found GPU0 NVIDIA GeForce GTX 1060 which is of cuda capability 6.1.
    Minimum and Maximum cuda capability supported by this version of PyTorch is
    (7.0) - (12.0))
- [ ] 推理

## 🚀 环境准备与安装

本项目使用 [uv](https://github.com/astral-sh/uv) 作为包管理工具。请确保你的环境中已安装 Python >= 3.13。

### **1. 克隆仓库**

```bash
git clone https://github.com/your-username/ml-j.git
cd ml-j
```

### **2. 安装 uv**

如果你还没有安装 `uv`，可以通过 `pip` 进行安装：

```bash
pip install uv
```

### **3. 创建并激活虚拟环境**

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### **4. 安装项目依赖**

使用 `uv` 安装所有必要的依赖包。

```bash
# 安装项目和开发依赖
uv pip install -e .[dev]
```

## 📦 使用方法

### 训练

你可以通过 `pyproject.toml` 中定义的脚本直接启动训练。

```bash
train --config config.yaml
```

请根据需要修改 `config.yaml` 文件中的配置。

### 推理

运行 `predict.py` 脚本来进行推理：

```bash
python src/mini_ml/core/predict.py --config config.yaml --image path/to/your/image.jpg
```

*(请注意: 上述命令是一个示例，具体参数请参照 `predict.py` 的实现)*

### Jupyter Notebook

项目根目录下的 `main.ipynb` 提供了一个交互式的示例，你可以通过它来了解项目的核心功能。

## 📝 配置

项目的主要配置在根目录的 `config.yaml` 文件中进行。你可以根据自己的数据集和模型需求进行调整。

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。
