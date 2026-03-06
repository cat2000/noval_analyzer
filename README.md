# 📚 强势文化专家系统 (天道 RAG)

基于《遥远的救世主》深度定制的本地 RAG (检索增强生成) 助手。
结合本地大模型 (Ollama + Qwen) 与向量数据库，提供符合“强势文化”逻辑的深度问答。

## ✨ 功能特点
- **本地运行**：数据不出域，隐私安全。
- **流式响应**：打字机效果，实时感知生成进度。
- **动态反馈**：智能呼吸灯提示，清晰展示“检索 - 思考 - 生成”状态。
- **断点续传**：支持大规模文本的分片处理与进度保存。

## 🛠️ 环境准备

1. **安装 Ollama**
   - 下载地址: [https://ollama.com](https://ollama.com)
   - 拉取模型 (推荐 Qwen):
     ```bash
     ollama pull qwen:7b
     # 或者
     ollama pull qwen:14b
     ```

2. **安装 Python 依赖**
   ```bash
   pip install -r requirements.txt