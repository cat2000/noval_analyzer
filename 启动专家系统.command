#!/bin/bash

# 设置窗口标题
echo "正在启动强势文化专家系统 (混合检索 + 证据链增强版)..."

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
    echo "   python3 -m venv .venv"
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

# 🚀 强力修复版：强制升级核心包到最新版
echo "📦 正在强制更新 LangChain 核心组件 (确保版本 >= 0.3)..."

pip install --upgrade --quiet \
    "langchain>=0.3.0" \
    "langchain-core>=0.3.0" \
    "langchain-community>=0.3.0" \
    "langchain-text-splitters" \
    "langchain-ollama" \
    "langchain-huggingface" \
    chromadb \
    streamlit \
    pypdf \
    tiktoken \
    sentence-transformers \
    scikit-learn \
    rank_bm25

if [ $? -ne 0 ]; then
    echo "❌ 依赖库安装失败。请尝试手动运行: source .venv/bin/activate && pip install rank_bm25 ..."
    exit 1
fi
echo "✅ 依赖库准备就绪 (包含 BGE 嵌入、BM25 检索及证据链计算模块)。"

# 4. 检查 Ollama 服务
echo "🤖 检查 Ollama 服务状态..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  未检测到 Ollama 服务运行。"
    if command -v ollama &> /dev/null; then
        echo "   正在尝试后台启动 Ollama..."
        # macOS/Linux 通用后台启动方式
        nohup ollama serve > ollama.log 2>&1 &
        sleep 3
        if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
             echo "❌ 自动启动 Ollama 失败。请务必手动打开 'Ollama' 应用程序或运行 'ollama serve'。"
             # 不要直接退出，让用户有机会手动启动，或者继续尝试
             echo "   将在 5 秒后再次尝试检测..."
             sleep 5
             if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
                 exit 1
             fi
        else
            echo "   ✅ Ollama 服务已成功后台启动。"
        fi
    else
        echo "❌ 错误：未找到 Ollama 命令行工具。请先安装 Ollama (https://ollama.com)。"
        exit 1
    fi
else
    echo "✅ Ollama 服务运行中。"
fi

# 5. 检查 LLM 模型
echo "📥 检查 LLM 模型库..."
# 注意：如果您之前用的是 qwen2.5 或其他版本，请相应修改此处
MODEL_NAME="qwen3" 
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "   ⚠️  未检测到模型 '$MODEL_NAME'。"
    echo "   正在下载 $MODEL_NAME 模型 (首次需要较长时间，请耐心等待)..."
    ollama pull $MODEL_NAME
else
    echo "   ✅ $MODEL_NAME 模型已就绪。"
fi

# 6. 配置 HuggingFace 镜像 (关键步骤)
# 在中国大陆地区，强烈建议使用镜像加速 BGE 模型的下载
export HF_ENDPOINT=https://hf-mirror.com
echo "🌐 已配置 HuggingFace 镜像加速 (hf-mirror.com)"
echo "💡 提示：Embedding 模型 (bge-large-zh) 将在程序首次运行时自动下载。"

# 7. 启动 Streamlit
echo "🚀 正在启动 Web 界面..."
echo "----------------------------------------"
echo "页面将在浏览器中自动打开。"
echo "当前 Python : $(which python)"
echo "Embedding   : BAAI/bge-large-zh (语义检索)"
echo "Retriever   : Hybrid (BM25 关键词 + Vector 语义)"
echo "LLM         : Ollama/$MODEL_NAME"
echo "Features    : ✅ 证据链可视化 | ✅ 章节元数据过滤"
echo "----------------------------------------"
echo "如需停止系统，请在此窗口按 Ctrl+C。"
echo ""

# 运行 Streamlit
# 使用 server.headless true 防止在某些环境下尝试打开浏览器失败
streamlit run app.py --server.headless true

# 保持窗口 (可选，如果是在终端直接运行，Ctrl+C 就会停，这里是为了脚本执行完不立即关闭终端)
# 如果是双击 .sh 文件运行，这行很有用；如果是终端运行，可以注释掉
echo ""
read -p "服务已停止。按回车键退出..."