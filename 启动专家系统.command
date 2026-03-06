#!/bin/bash

# 设置窗口标题
echo "正在启动强势文化专家系统 (专用虚拟环境版)..."

# 获取当前脚本所在目录，确保在正确的路径下运行
cd "$(dirname "$0")"

# ================= 配置区域 =================
# 指定虚拟环境激活脚本的路径
VENV_ACTIVATE="./.venv/bin/activate"
# 指定虚拟环境中的 python 路径
VENV_PYTHON="./.venv/bin/python3"
# ===========================================

# 1. 检查虚拟环境是否存在
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ 错误：未找到虚拟环境文件夹 (.venv)！"
    echo "   请在该项目目录下先运行以下命令创建环境："
    echo "   /opt/homebrew/bin/python3.11 -m venv .venv"
    echo ""
    echo "按任意键退出..."
    read -n 1
    exit 1
fi

echo "✅ 发现虚拟环境，正在激活..."

# 2. 激活虚拟环境
source "$VENV_ACTIVATE"

# 验证是否激活成功 (检查 python 路径是否指向 .venv)
CURRENT_PYTHON=$(which python)
if [[ "$CURRENT_PYTHON" != *".venv"* ]]; then
    echo "❌ 错误：虚拟环境激活失败！当前 python 路径为: $CURRENT_PYTHON"
    exit 1
fi
echo "✅ 虚拟环境激活成功: $CURRENT_PYTHON"

# 3. 自动安装/更新依赖库
echo "📦 正在检查并安装必要的依赖库 (首次运行可能需要几分钟)..."
# 使用虚拟环境内的 pip
pip install --quiet --upgrade pip
pip install --quiet langchain langchain-text-splitters langchain-community langchain-ollama chromadb streamlit pypdf tiktoken langchain-core

if [ $? -ne 0 ]; then
    echo "❌ 依赖库安装失败。请尝试手动运行: source .venv/bin/activate && pip install ..."
    exit 1
fi
echo "✅ 依赖库准备就绪。"

# 4. 检查 Ollama 服务
echo "🤖 检查 Ollama 服务状态..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  未检测到 Ollama 服务运行。"
    if command -v ollama &> /dev/null; then
        echo "   正在尝试后台启动 Ollama..."
        # Mac 上通常 ollama 是以 App 形式运行在后台，如果命令行工具存在但服务没起，尝试启动
        # 注意：Mac 版 Ollama 通常建议手动打开 App，这里尝试命令行启动作为备选
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

# 5. 检查并下载模型 (使用虚拟环境外的 ollama 命令，因为 ollama 是系统级工具)
# 注意：ollama 命令通常不在虚拟环境中，所以直接调用系统 PATH 中的 ollama
echo "📥 检查模型库..."
if ! ollama list | grep -q "qwen3"; then
    echo "   正在下载 Qwen 模型 (首次需要较长时间，请耐心等待)..."
    ollama pull qwen3
fi

if ! ollama list | grep -q "nomic-embed-text"; then
    echo "   正在下载 nomic-embed-text 嵌入模型..."
    ollama pull nomic-embed-text
fi
echo "✅ 模型检查完毕。"

# 6. 启动 Streamlit
echo "🚀 正在启动 Web 界面..."
echo "----------------------------------------"
echo "页面将在浏览器中自动打开。"
echo "当前使用的 Python: $(which python)"
echo "如需停止系统，请在此窗口按 Ctrl+C。"
echo "----------------------------------------"

# 使用虚拟环境中的 streamlit 运行
streamlit run app.py

# 保持窗口（如果意外退出）
echo ""
read -p "服务已停止。按回车键退出..."