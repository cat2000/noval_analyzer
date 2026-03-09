#!/bin/bash

# 设置窗口标题
echo "正在启动强势文化专家系统 (Self-RAG + Rerank 进阶版)..." # ✅ [修改] 标题更新
echo "------------------------------------------------"

# 获取当前脚本所在目录
cd "$(dirname "$0")"

# ================= 配置区域 =================
VENV_ACTIVATE="./.venv/bin/activate"
MODEL_NAME="qwen3" 
EVAL_MODEL_NAME="qwen3:0.6b"
# 新增：HuggingFace 模型配置 (用于日志显示，实际下载由代码控制)
EMBED_MODEL="BAAI/bge-large-zh"
RERANK_MODEL="BAAI/bge-reranker-large"
# ===========================================

# 1. 检查虚拟环境
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ 错误：未找到虚拟环境 (.venv)！"
    echo "💡 请先运行: python -m venv .venv"
    exit 1
fi

echo "✅ 激活虚拟环境..."
source "$VENV_ACTIVATE"

CURRENT_PYTHON=$(which python)
if [[ "$CURRENT_PYTHON" != *".venv"* ]]; then
    echo "❌ 虚拟环境激活失败！"
    exit 1
fi

# 2. 安装依赖 (静默模式)
echo "📦 检查并升级依赖库..."
pip install --quiet --upgrade pip

# ✅ [修改] 确保 sentence-transformers 和相关库是最新的，以支持 CrossEncoder
pip install --upgrade --quiet \
    "langchain>=0.3.0" "langchain-core>=0.3.0" "langchain-community>=0.3.0" \
    "langchain-text-splitters" "langchain-ollama" "langchain-huggingface" \
    "chromadb" "streamlit" "sentence-transformers>=0.29.0" "scikit-learn" "rank_bm25" "requests" "torch"

# 3. 检查 Ollama
echo "🤖 检查 Ollama 服务..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "   ⚠️  Ollama 未运行，尝试启动..."
    if command -v ollama &> /dev/null; then
        nohup ollama serve > ollama.log 2>&1 &
        sleep 4
        if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
            echo "❌ Ollama 启动失败，请手动运行 'ollama serve'。"
            exit 1
        fi
        echo "   ✅ Ollama 服务已启动。"
    else
        echo "❌ 未找到 ollama 命令。"
        exit 1
    fi
else
    echo "✅ Ollama 服务运行中。"
fi

# 4. 检查 LLM 模型 (Ollama)
echo "📥 检查 Ollama 模型库..."
for model in "$MODEL_NAME" "$EVAL_MODEL_NAME"; do
    if ! ollama list | grep -q "^$model\s"; then
        echo "   ⚠️  下载模型 $model (这可能需要几分钟)..."
        ollama pull $model
    else
        echo "   ✅ $model 已就绪。"
    fi
done

# 5. 配置 HuggingFace 镜像 (关键！用于加速 Embedding 和 Rerank 模型下载)
# ✅ [修改] 强调这一步对新版架构的重要性
echo "🌐 配置 HuggingFace 镜像源 (加速 BGE 模型下载)..."
export HF_ENDPOINT=https://hf-mirror.com
echo "   ✅ 已设置 HF_ENDPOINT=$HF_ENDPOINT"

# 6. 启动 Streamlit
echo "🚀 正在启动 Web 界面..."
echo "----------------------------------------"
echo "📊 架构配置:"
echo "   - 生成/重写 LLM : $MODEL_NAME"
echo "   - 评估 LLM      : $EVAL_MODEL_NAME"
echo "   - Embedding     : $EMBED_MODEL (自动下载)"
echo "   - Rerank        : $RERANK_MODEL (自动下载)"
echo "----------------------------------------"

URL="http://localhost:8501"

# 启动命令
STREAMLIT_CMD="streamlit run app.py --server.headless true --server.address localhost --server.port 8501"

# 后台启动
$STREAMLIT_CMD &
STREAMLIT_PID=$!

# 等待服务就绪 (最多等待 15 秒，因为首次加载 Rerank 模型可能稍慢)
echo "⏳ 等待服务启动 (首次运行需下载 BGE 模型，请稍候)..."
for i in {1..15}; do
    if curl -s $URL &> /dev/null; then
        echo "✅ 服务已就绪!"
        break
    fi
    sleep 1
done

# 打开浏览器
if curl -s $URL &> /dev/null; then
    echo "🌐 正在打开浏览器..."
    
    if command -v open &> /dev/null; then
        open "$URL"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$URL"
    elif command -v start &> /dev/null; then
        start "$URL"
    else
        echo "💡 请手动在浏览器访问: $URL"
    fi
else
    echo "⚠️  服务启动超时。"
    echo "💡 首次运行可能需要下载 BGE 模型文件 (约 1GB+)，请耐心等待终端输出。"
    echo "   或者手动访问: $URL"
fi

echo ""
echo "✅ 系统运行中 (PID: $STREAMLIT_PID)"
echo "按 Ctrl+C 停止服务..."

# 捕获 Ctrl+C 以清理进程
trap "kill $STREAMLIT_PID 2>/dev/null; echo ''; echo '服务已停止。'; exit" INT TERM

# 等待主进程
wait $STREAMLIT_PID