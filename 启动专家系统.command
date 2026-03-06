#!/bin/bash

# 设置窗口标题
echo "正在启动强势文化专家系统 (BGE-ZH + 证据链增强版)..."

# 获取当前脚本所在目录，确保在正确的路径下运行
cd "$(dirname "$0")"

# ================= 配置区域 =================
VENV_ACTIVATE="./.venv/bin/activate"
VENV_PYTHON="./.venv/bin/python3"
# ===========================================

# 1. 检查虚拟环境是否存在
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ 错误：未找到虚拟环境文件夹 (.venv)！"
    echo "   请在该项目目录下先运行以下命令创建环境："
    echo "   /opt/homebrew/bin/python3.11 -m venv .venv"
    echo ""
    read -n 1
    exit 1
fi

echo "✅ 发现虚拟环境，正在激活..."

# 2. 激活虚拟环境
source "$VENV_ACTIVATE"

# 验证是否激活成功
CURRENT_PYTHON=$(which python)
if [[ "$CURRENT_PYTHON" != *".venv"* ]]; then
    echo "❌ 错误：虚拟环境激活失败！当前 python 路径为: $CURRENT_PYTHON"
    exit 1
fi
echo "✅ 虚拟环境激活成功: $CURRENT_PYTHON"

# 3. 自动安装/更新依赖库
echo "📦 正在检查并安装必要的依赖库..."
pip install --quiet --upgrade pip

# ✅ 关键修改：添加 scikit-learn 用于计算证据链相似度得分
# 同时确保 langchain-huggingface 和 sentence-transformers 已安装
pip install --quiet \
    langchain \
    langchain-text-splitters \
    langchain-community \
    langchain-ollama \
    langchain-huggingface \
    chromadb \
    streamlit \
    pypdf \
    tiktoken \
    langchain-core \
    sentence-transformers \
    scikit-learn  # <-- 新增：用于 cosine_similarity

if [ $? -ne 0 ]; then
    echo "❌ 依赖库安装失败。请尝试手动运行: source .venv/bin/activate && pip install ..."
    exit 1
fi
echo "✅ 依赖库准备就绪 (包含 BGE 支持与证据链计算模块)。"

# 4. 检查 Ollama 服务
echo "🤖 检查 Ollama 服务状态..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  未检测到 Ollama 服务运行。"
    if command -v ollama &> /dev/null; then
        echo "   正在尝试后台启动 Ollama..."
        ollama serve &
        sleep 3
        if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
             echo "❌ 自动启动 Ollama 失败。请务必手动打开 'Ollama' 应用程序。"
             exit 1
        fi
        echo "   Ollama 服务已启动。"
    else
        echo "❌ 错误：未找到 Ollama 命令行工具。请先安装 Ollama。"
        exit 1
    fi
else
    echo "✅ Ollama 服务运行中。"
fi

# 5. 检查 LLM 模型
echo "📥 检查 LLM 模型库..."
if ! ollama list | grep -q "qwen3"; then
    echo "   正在下载 Qwen 模型 (首次需要较长时间，请耐心等待)..."
    ollama pull qwen3
else
    echo "   ✅ Qwen3 模型已就绪。"
fi

# 6. 配置 HuggingFace 镜像 (关键步骤)
# 在启动 Python 前强制设置，确保 BGE 模型下载不超时
export HF_ENDPOINT=https://hf-mirror.com
echo "🌐 已配置 HuggingFace 镜像加速 (hf-mirror.com)"
echo "💡 提示：Embedding 模型 (bge-large-zh) 将在程序首次运行时自动下载。"

# 7. 启动 Streamlit
echo "🚀 正在启动 Web 界面..."
echo "----------------------------------------"
echo "页面将在浏览器中自动打开。"
echo "当前 Python: $(which python)"
echo "Embedding : BAAI/bge-large-zh (本地 CPU/GPU 推理)"
echo "LLM       : Ollama/qwen3"
echo "功能      : ✅ 证据链可视化已启用"
echo "----------------------------------------"
echo "如需停止系统，请在此窗口按 Ctrl+C。"
echo ""

# 运行 Streamlit
streamlit run app.py

# 保持窗口
echo ""
read -p "服务已停止。按回车键退出..."