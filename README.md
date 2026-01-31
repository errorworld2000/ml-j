# mini-ml

一个用于模型重写的小型项目。

## ✨ 功能清单

- [x] 训练
- [x] 推理

## 🚀 环境准备与安装

本项目使用 [uv](https://github.com/astral-sh/uv) 作为包管理工具。请确保你的环境中已安装 Python >= 3.13。

### **1. 克隆仓库**

```bash
git clone https://github.com/your-username/ml-j.git
cd ml-j
```

### **2. 安装 uv**

官网安装

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
uv sync
uv pip install -e .[dev]
```

## 📦 使用方法

### 训练

```bash
uv run train
```

### 推理

```shell
uv run predict
```

## 📂 项目结构

关于代码结构的详细分析，请查阅 [STRUCTURE.md](docs/STRUCTURE.md)。

## 📝 配置

项目的主要配置在根目录的 `config.yaml` 文件中进行。你可以根据自己的数据集和模型需求进行调整。

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。
