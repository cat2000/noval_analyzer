import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 路径配置 ---
DATA_FILE = os.path.join(BASE_DIR, "data", "遥远的救世主.txt")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_data")

# --- 模型配置 ---
# 推荐：qwen2.5:7b 或 qwen2.5:14b (逻辑强，中文好)
LLM_MODEL = "qwen3" 
# 推荐：BAAI/bge-m3 (中文嵌入效果极佳)
EMBEDDING_MODEL = "BAAI/bge-m3"

# --- RAG 策略配置 ---
# 合理拆分：600字符块，150重叠，保证语义连贯且不过大
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
SEARCH_K = 5  # 每次检索返回的片段数

# --- 系统信息 ---
SYSTEM_TITLE = "《遥远的救世主》· 天道分析系统"
SYSTEM_ICON = "⚖️"