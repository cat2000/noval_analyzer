#!/bin/bash

# 设置窗口标题
echo "🚀 正在启动强势文化专家系统 (Self-RAG + CoT + Rerank 进阶版)..." 
echo "------------------------------------------------"

# 获取当前脚本所在目录
cd "$(dirname "$0")"

# ================= 配置区域 =================
VENV_ACTIVATE="./.venv/bin/activate"
MODEL_NAME="qwen3" 
EVAL_MODEL_NAME="qwen3:0.6b"
# 模型配置 (用于日志显示)
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
# [修改] 去掉了 --upgrade 参数，如果包已存在则直接跳过，大幅加快速度
echo "📦 检查并安装缺失依赖库..."
pip install --quiet pip

# ✅ [修改] 添加了 jieba 用于关键词快速过滤
# 注意：去掉了 --upgrade，只有缺失时才会安装
pip install --quiet \
    "langchain>=0.3.0" "langchain-core>=0.3.0" "langchain-community>=0.3.0" \
    "langchain-text-splitters" "langchain-ollama" "langchain-huggingface" \
    "chromadb" "streamlit" "sentence-transformers>=0.29.0" "scikit-learn" \
    "rank_bm25" "requests" "torch" "jieba"

# 3. 检查 Ollama
echo "🤖 检查 Ollama 服务..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "   ⚠️  Ollama 未运行，尝试启动..."
    if command -v ollama &> /dev/null; then
        # 确保后台启动不占用当前终端输出
        nohup ollama serve > ollama.log 2>&1 &
        sleep 4
        if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
            echo "❌ Ollama 启动失败，请手动运行 'ollama serve' 查看错误日志。"
            exit 1
        fi
        echo "   ✅ Ollama 服务已启动。"
    else
        echo "❌ 未找到 ollama 命令，请先安装 Ollama。"
        exit 1
    fi
else
    echo "✅ Ollama 服务运行中。"
fi

# 4. 检查 LLM 模型 (Ollama)
echo "📥 检查 Ollama 模型库..."
for model in "$MODEL_NAME" "$EVAL_MODEL_NAME"; do
    # ✅ [优化] 放宽 grep 匹配规则，防止因空格导致的匹配失败
    if ! ollama list | grep -q "$model"; then
        echo "   ⚠️  下载模型 $model (这可能需要几分钟)..."
        ollama pull "$model"
    else
        echo "   ✅ $model 已就绪。"
    fi
done

# 5. 配置 HuggingFace 镜像 (关键！用于加速 Embedding 和 Rerank 模型下载)
echo "🌐 配置 HuggingFace 镜像源 (加速 BGE 模型下载)..."
export HF_ENDPOINT=https://hf-mirror.com
echo "   ✅ 已设置 HF_ENDPOINT=$HF_ENDPOINT"

# 6. 启动 Streamlit
echo "🚀 正在启动 Web 界面..."
echo "----------------------------------------"
echo "📊 架构配置:"
echo "   - 生成/重写 LLM : $MODEL_NAME"
echo "   - 评估 LLM      : $EVAL_MODEL_NAME (CoT 增强)"
echo "   - Embedding     : $EMBED_MODEL (自动下载)"
echo "   - Rerank        : $RERANK_MODEL (自动下载)"
echo "   - 分词工具      : jieba (关键词预检)"
echo "----------------------------------------"

URL="http://localhost:8501"

# [新增] 检查是否已经运行，如果已运行则直接跳过启动
if curl -s $URL &> /dev/null; then
    echo "✅ 检测到服务已在运行 ($URL)，跳过启动步骤。"
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
    echo ""
    echo "✅ 系统运行中。按 Ctrl+C 退出脚本 (服务将在后台继续运行)。"
    # 捕获 Ctrl+C 以提示
    trap "echo ''; echo '🛑 脚本已停止 (服务仍在后台运行)。'; exit" INT TERM
    wait
    exit 0
fi

# 启动命令
STREAMLIT_CMD="streamlit run app.py --server.headless true --server.address localhost --server.port 8501"

# 后台启动
$STREAMLIT_CMD > streamlit.log 2>&1 &
STREAMLIT_PID=$!

# ✅ [优化] 延长等待时间至 40 秒，因为首次加载 Rerank 大模型较慢
WAIT_TIME=40
echo "⏳ 等待服务启动 (首次运行需下载/加载 BGE 模型，请稍候)..."
for i in $(seq 1 $WAIT_TIME); do
    if curl -s $URL &> /dev/null; then
        echo "✅ 服务已就绪! (耗时 ${i}s)"
        break
    fi
    # 每 5 秒给一个提示，避免用户以为卡死
    if [ $((i % 5)) -eq 0 ]; then
        echo "   ... 正在加载重排序模型 (这可能比较慢) ..."
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
    echo ""
    echo "✅ 系统运行中 (PID: $STREAMLIT_PID)"
    echo "📝 日志文件: streamlit.log"
    echo "按 Ctrl+C 停止服务..."
else
    echo "⚠️  服务启动超时 (${WAIT_TIME}s)。"
    echo "💡 可能原因："
    echo "   1. 首次下载 BGE 模型文件较大 (约 1.3GB)，网速较慢。"
    echo "   2. 显存/内存不足导致模型加载失败。"
    echo "👉 请查看 'streamlit.log' 获取详细错误信息。"
    echo "   或者手动访问: $URL (如果服务其实已经启动只是慢了点)"
fi

# 捕获 Ctrl+C 以清理进程
trap "kill $STREAMLIT_PID 2>/dev/null; echo ''; echo '🛑 服务已停止。'; exit" INT TERM

# 等待主进程
wait $STREAMLIT_PID