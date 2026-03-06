#!/bin/bash

# 1. 获取脚本所在目录并进入，确保路径正确
cd "$(dirname "$0")"

echo "=========================================="
echo "  🚀 正在启动《遥远的救世主》分析系统"
echo "  当前目录: $(pwd)"
echo "=========================================="

# 2. 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "❌ 错误：未找到 '.venv' 文件夹。"
    echo "💡 请确认你已在当前目录下创建并配置了虚拟环境。"
    echo "   (例如: python3 -m venv .venv)"
    read -p "按回车键退出..."
    exit 1
fi

echo "✅ 发现虚拟环境，正在激活..."

# 3. 激活虚拟环境
# 注意：在脚本中使用 source 需要特殊处理，或者直接调用 venv 中的 python
source .venv/bin/activate

# 4. 再次检查 streamlit 是否已安装
if ! command -v streamlit &> /dev/null; then
    echo "⚠️ 虚拟环境中未找到 streamlit，正在尝试安装..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败，请检查 requirements.txt 或网络连接。"
        read -p "按回车键退出..."
        exit 1
    fi
fi

echo "✅ 环境准备就绪，正在启动服务..."
echo ""

# 5. 启动 Streamlit
# --server.headless false: 强制自动打开浏览器
# --browser.gatherUsageStats false: 关闭数据统计
streamlit run app.py --server.headless false --browser.gatherUsageStats false --server.port 8501

# 6. 窗口保持
echo ""
echo "🛑 服务已停止。按回车键关闭此窗口..."
read