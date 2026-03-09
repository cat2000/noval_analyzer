#!/bin/bash

# 设置窗口标题
echo "正在启动强势文化专家系统 (CRAG Qwen3 双模型加速版)..."
echo "------------------------------------------------"

# 获取当前脚本所在目录
cd "$(dirname "$0")"

# ================= 配置区域 =================
VENV_ACTIVATE="./.venv/bin/activate"
MODEL_NAME="qwen3" 
EVAL_MODEL_NAME="qwen3:0.6b"
# ===========================================

# 1. 检查虚拟环境
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ 错误：未找到虚拟环境 (.venv)！"
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
echo "📦 检查依赖库..."
pip install --quiet --upgrade pip
pip install --upgrade --quiet \
    "langchain>=0.3.0" "langchain-core>=0.3.0" "langchain-community>=0.3.0" \
    "langchain-text-splitters" "langchain-ollama" "langchain-huggingface" \
    "chromadb" "streamlit" "sentence-transformers" "scikit-learn" "rank_bm25" "requests"

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

# 4. 检查模型
echo "📥 检查模型库..."
for model in "$MODEL_NAME" "$EVAL_MODEL_NAME"; do
    if ! ollama list | grep -q "^$model\s"; then
        echo "   ⚠️  下载模型 $model ..."
        ollama pull $model
    else
        echo "   ✅ $model 已就绪。"
    fi
done

# 5. 配置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 6. 启动 Streamlit
echo "🚀 正在启动 Web 界面..."
echo "----------------------------------------"
echo "📊 配置: $MODEL_NAME + $EVAL_MODEL_NAME"
echo "----------------------------------------"

URL="http://localhost:8501"

# ✅ [修复核心] 
# 1. 使用 --server.headless true 禁止 Streamlit 自身尝试打开浏览器 (防止双重打开)
# 2. 脚本在检测到服务启动后，统一由脚本尝试打开一次浏览器
STREAMLIT_CMD="streamlit run app.py --server.headless true --server.address localhost --server.port 8501"

# 后台启动
$STREAMLIT_CMD &
STREAMLIT_PID=$!

# 等待服务就绪 (最多等待 10 秒)
echo "⏳ 等待服务启动..."
for i in {1..10}; do
    if curl -s $URL &> /dev/null; then
        echo "✅ 服务已就绪!"
        break
    fi
    sleep 1
done

# ✅ [单一入口] 仅由脚本尝试打开浏览器一次
if curl -s $URL &> /dev/null; then
    echo "🌐 正在打开浏览器..."
    
    # 尝试打开浏览器
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
    echo "⚠️  服务启动超时，请手动检查日志或访问: $URL"
fi

echo ""
echo "✅ 系统运行中 (PID: $STREAMLIT_PID)"
echo "按 Ctrl+C 停止服务..."

# 捕获 Ctrl+C 以清理进程
trap "kill $STREAMLIT_PID 2>/dev/null; echo ''; echo '服务已停止。'; exit" INT TERM

# 等待主进程
wait $STREAMLIT_PID