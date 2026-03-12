import streamlit as st
from rag_engine import RAGEngine
import os
import time
import textwrap

# ==========================================
# 1. 配置与常量区 (Configuration & Constants)
# ==========================================
PAGE_CONFIG = {
    "page_title": "《遥远的救世主》读书小精灵",
    "page_icon": "📚",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

PARAM_CONFIG = {
    "rewrite_mode": {"label": "改写策略", "type": "select", "options": ["direct", "light", "deep"], "format_map": {"direct": "直连", "light": "轻量", "deep": "深度"}, "default": "deep", "desc_short": "Query 重写模式", "home_node_id": "rewrite"},
    "multi_query_count": {"label": "发散路数", "type": "slider", "min": 0, "max": 5, "step": 1, "default": 1, "desc_short": "多路查询变体数量", "home_node_id": "multi_query"},
    "top_k_initial": {"label": "初召 TopK", "type": "slider", "min": 5, "max": 50, "step": 5, "default": 8, "desc_short": "混合召回数量 (BM25+向量)", "home_node_id": "retrieval"},
    "rerank_threshold": {"label": "重排门槛", "type": "slider", "min": 0.0, "max": 0.9, "step": 0.05, "default": 0.5, "desc_short": "低于此分数的片段将被丢弃", "home_node_id": "rerank"},
    "max_self_rag_attempts": {"label": "最大重试", "type": "slider", "min": 0, "max": 3, "step": 1, "default": 1, "desc_short": "检索质量不佳时自动重试", "home_node_id": "eval"},
    "top_k_final": {"label": "终选 TopK", "type": "slider", "min": 1, "max": 10, "step": 1, "default": 3, "desc_short": "最终送入上下文的最大片段数", "home_node_id": "generation"}
}

PIPELINE_NODES = [
    {"id": "rewrite", "title": "语义重写", "icon": "🧠", "controlled_by": ["rewrite_mode"], "desc": "Query 转化", "stage_key": "Query_Rewrite"},
    {"id": "multi_query", "title": "多路发散", "icon": "🔀", "controlled_by": ["multi_query_count", "rewrite_mode"], "desc": "多角度变体", "stage_key": "MultiQuery_Gen"},
    {"id": "retrieval", "title": "混合检索", "icon": "📚", "controlled_by": ["top_k_initial"], "desc": "向量 + 关键词", "stage_key": "Parallel_Retrieval"},
    {"id": "rerank", "title": "重排序", "icon": "⚖️", "controlled_by": ["top_k_initial", "rerank_threshold", "top_k_final"], "desc": "精细打分", "stage_key": "Rerank"},
    {"id": "eval", "title": "质量评估", "icon": "🛡️", "controlled_by": ["rerank_threshold", "max_self_rag_attempts"], "desc": "相关性判断 (含重试)", "stage_key": "Eval_Phase"}, 
    {"id": "generation", "title": "文章生成", "icon": "✍️", "controlled_by": ["top_k_final"], "desc": "LLM 撰写", "stage_key": "LLM_Generation"}
]

TECH_STACK = [
    {"label": "大语言模型", "value": "Qwen3 (Ollama)", "desc": "本地部署的高性能开源模型，负责推理与生成。", "badge": "LLM"},
    {"label": "嵌入模型", "value": "BGE-M3", "desc": "支持多语言、长上下文的稠密向量模型。", "badge": "Embedding"},
    {"label": "重排序模型", "value": "BGE-Reranker-v2-m3", "desc": "Cross-Encoder 架构，精准评估 Query-Doc 相关性。", "badge": "Rerank"},
    {"label": "向量数据库", "value": "ChromaDB / FAISS", "desc": "高性能向量存储与相似度检索引擎。", "badge": "Vector DB"},
    {"label": "应用框架", "value": "Streamlit + LangChain", "desc": "快速构建数据驱动的 Web 应用。", "badge": "Framework"},
    {"label": "知识库源", "value": "《遥远的救世主》.txt", "desc": "豆豆著，关于文化属性与强势文化的经典著作。", "badge": "Data"}
]

# ==========================================
# 2. 样式定义 (Styles) - ✅ 已移除所有强制固定高度的 CSS
# ==========================================
def get_custom_css():
    return """
    <style>
        /* 全局背景与字体 */
        .stApp { 
            background-color: #fcfbf9; 
            background-image: linear-gradient(to bottom, #fcfbf9 0%, #f7f5f0 100%); 
            /* ✅ 删除了 overflow: hidden, height: 100vh, display: flex 等强制固定样式 */
            /* 让页面自然滚动 */
        }
        body { 
            font-family: 'Songti SC', 'STSong', 'Noto Serif SC', 'SimSun', serif; 
            color: #2c2c2c; 
            margin: 0; 
            padding: 0;
        }
        
        /* 主容器：恢复默认行为，允许自然高度 */
        .block-container { 
            padding-top: 1rem !important; 
            padding-bottom: 2rem !important; 
            max-width: 2000px !important; /* 限制最大宽度，避免太宽 */
            /* ✅ 删除了 height, display: flex, overflow: hidden 等 */
        }

        /* 面板通用样式：保留美观，但移除高度限制 */
        .sidebar-panel { 
            background: #fff; 
            border-radius: 12px; 
            border: 1px solid #efebe9; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
            padding: 15px; 
            /* ✅ 删除了 height: 100%, overflow-y: auto */
            /* 让面板内容自然撑开高度 */
            margin-bottom: 20px; /* 增加底部间距 */
        }
        
        /* 聊天区域样式优化 */
        .chat-section { 
            background: #fff; 
            border-radius: 12px; 
            border: 1px solid #e0e0e0; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.03); 
            padding: 10px 0;
            margin-bottom: 15px;
        }
        
        /* 聊天记录滚动区：不再强制固定高度，让它随内容增长 */
        .chat-history { 
            /* ✅ 删除了 flex: 1, min-height: 0, overflow-y: auto */
            /* 让消息列表自然向下延伸 */
            padding: 10px 15px; 
        }
        
        /* 底部控制台：不再限制最大高度，允许展开 */
        .bottom-console { 
            background: #fff; 
            border-radius: 12px; 
            border: 1px solid #e0e0e0; 
            padding: 10px 15px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.03); 
            /* ✅ 删除了 max-height, overflow-y: auto */
            display: flex; 
            flex-direction: column; 
            gap: 5px; 
        }

        /* 滚动条美化 (Webkit) */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb { background: #d7ccc8; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #bcaaa4; }

        /* 其他原有样式保持不变... */
        .panel-title { font-size: 1.1rem; font-weight: bold; color: #5d4037; margin-bottom: 15px; border-left: 4px solid #8d6e63; padding-left: 10px; display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
        .tech-item { margin-bottom: 8px; padding: 6px; background: #f9f9f9; border-radius: 8px; border: 1px solid #eee; }
        .tech-label { font-size: 0.75rem; color: #90a4ae; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
        .tech-value { font-size: 0.95rem; color: #2c3e50; font-weight: bold; font-family: 'Consolas', monospace; }
        .tech-desc { font-size: 0.8rem; color: #546e7a; margin-top: 4px; line-height: 1.4; }
        .badge-tech { display: inline-block; background: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-right: 5px; }
        
        /* 证据列表样式 */
        .evidence-list { display: flex; flex-direction: column; gap: 12px; }
        .evidence-card { padding: 12px; border-radius: 8px; background: #fafafa; border: 1px solid #eee; transition: all 0.3s; }
        .evidence-card.cited { background: #fff8e1; border-color: #ffcc80; box-shadow: 0 2px 6px rgba(255, 179, 0, 0.1); transform: translateX(2px); }
        .evidence-card:not(.cited) { opacity: 0.75; filter: grayscale(0.2); }
        .evidence-meta { font-size: 0.75rem; color: #8d6e63; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }
        .evidence-text { font-size: 0.85rem; color: #37474f; line-height: 1.6; font-family: 'Songti SC', serif; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .badge-cited { background: #ffb74d; color: #fff; padding: 1px 5px; border-radius: 3px; font-size: 0.65rem; font-weight: bold; }
        .badge-uncited { background: #cfd8dc; color: #fff; padding: 1px 5px; border-radius: 3px; font-size: 0.65rem; }
        .empty-state { text-align: center; color: #90a4ae; font-style: italic; padding: 40px 10px; font-size: 0.9rem; display: flex; flex-direction: column; justify-content: center; }
        
        /* 头部样式 */
        .main-header { text-align: center; padding: 20px 0 30px 0; flex-shrink: 0; }
        .main-header h1 { font-family: 'Songti SC', serif; font-weight: bold; color: #2c3e50; letter-spacing: 2px; margin-bottom: 5px !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); font-size: 2rem; }
        .subtitle { color: #78909c; font-size: 1.1rem; font-style: italic; }
        
        /* 聊天输入区 */
        .chat-input-area { padding: 10px 15px; border-top: 1px solid #eee; background: #fcfbf9; z-index: 100; }
        .stChatMessage { padding: 8px 0; border-bottom: 1px solid #f5f5f5; }
        .stChatMessage:last-child { border-bottom: none; }
        
        /* 监控面板样式 */
        .pipeline-container { display: flex; flex-wrap: nowrap; gap: 8px; justify-content: stretch; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #eee; width: 100%; }
        .stColumn {
            display: flex !important;
            flex-direction: column !important;
            min-width: 0 !important;
        }        
        .node-card { width: 100%; padding: 8px 4px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.04); transition: all 0.3s; border-top: 3px solid transparent; box-sizing: border-box; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .node-card.bottleneck { border-color: #d84315; border-top: 3px solid #d84315; background-color: #fffaf9; box-shadow: 0 6px 15px rgba(216, 67, 21, 0.15); animation: gentle-pulse 3s infinite; }
        @keyframes gentle-pulse { 0% { transform: translateY(0); } 50% { transform: translateY(-2px); } 100% { transform: translateY(0); } }
        .node-icon { font-size: 1.3rem; margin-bottom: 2px; line-height: 1; }
        .node-title { font-weight: bold; color: #2c3e50; font-size: 0.75rem; margin-bottom: 2px; line-height: 1.2; }
        .node-time { font-size: 0.55rem; color: #90a4ae; font-family: 'Consolas', monospace; background: #f5f7fa; padding: 1px 3px; border-radius: 4px; line-height: 1; }
        .node-time.slow { color: #d84315; background: #ffebee; font-weight: bold; }
        .optimization-tip { margin-top: 2px; font-size: 0.55rem; color: #d84315; background: #fff3e0; padding: 1px 3px; border-radius: 3px; width: 100%; box-sizing: border-box; text-align: center; border: 1px dashed #ffccbc; line-height: 1.2; }
        .param-control-area { margin-top: 2px; margin-bottom: 2px; width: 100%; display: flex; flex-direction: column; align-items: center; }
        .param-label-mini { font-size: 0.6rem; color: #546e7a; font-weight: bold; margin-bottom: 1px; display: block; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; width: 100%; }
        .param-label-mini.urgent { color: #d84315; background: #fff3e0; padding: 1px 3px; border-radius: 2px; }
        .param-ref-tag { margin-top: 4px; font-size: 0.72rem; color: #78909c; background: #f5f7fa; padding: 2px 4px; border-radius: 4px; width: 100%; box-sizing: border-box; text-align: center; border: 1px dotted #cfd8dc; font-style: italic; line-height: 1.2; }
        
        /* 隐藏 Streamlit 默认元素 */
        .stSlider label, .stSelectbox label, .stNumberInput label { display: none !important; }
        .stSlider, .stSelectbox, .stNumberInput { margin-bottom: 0 !important; margin-top: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; min-height: auto !important; width: 100% !important; }
        .stSlider .stTextInput input, .stSelectbox .stTextInput input { font-size: 0.5rem !important; height: 15px !important; padding: 1px 3px !important; }
        div[data-baseweb="select"] span { font-size: 0.5rem !important; }
        div[data-baseweb="popover"] ul { font-size: 0.5rem !important; }
        div[data-baseweb="popover"] li { font-size: 0.5rem !important; padding: 4px 8px !important; }
        div[data-baseweb="popover"] li:hover { background: #f5f5f5 !important; }
        .stSlider div[data-baseweb="slider"] { margin-top: 2px !important; margin-bottom: 2px !important; }
        footer, header { visibility: hidden; height: 0; }
        
        /* 加载动画相关 */
        .loading-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh; color: #78909c; font-style: italic; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #8d6e63; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; margin-bottom: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fade-in { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """

def render_zen_thinking():
    """渲染问答过程中的禅意思考动画"""
    thinking_phrases = [
        "正在参悟文化属性...",
        "正在检索丁元英的思维轨迹...",
        "正在构建逻辑闭环...",
        "神即道，道法自然...",
        "正在权衡强势与弱势文化...",
        "正在翻阅《遥远的救世主》..."
    ]
    import random
    phrase = random.choice(thinking_phrases)
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; color: #8d6e63; font-style: italic; padding: 10px 0; animation: fade-in 0.5s ease-out;">
        <div style="width: 18px; height: 18px; border: 2px solid #d7ccc8; border-top: 2px solid #8d6e63; border-radius: 50%; animation: spin 1.5s linear infinite;"></div>
        <span style="font-family: 'Songti SC', serif; font-size: 0.95rem; letter-spacing: 1px;">{phrase}</span>
    </div>
    
    <style>
        @keyframes fade-in {{
            from {{ opacity: 0; transform: translateY(5px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """, unsafe_allow_html=True)

def render_elegant_loading():
    """渲染优雅的加载界面"""
    import random
    loading_quotes = [
        "正在唤醒丁元英的思维模型...",
        "正在构建文化属性向量空间...",
        "正在研读《遥远的救世主》全文...",
        "神即道，道法自然，如来... 正在加载中",
        "正在连接强势文化逻辑链..."
    ]
    quote = random.choice(loading_quotes)
    
    css_code = """
    <style>
        @keyframes progress-indeterminate {
            0% { transform: translateX(-100%) scaleX(0.2); }
            50% { transform: translateX(0%) scaleX(0.5); }
            100% { transform: translateX(100%) scaleX(0.2); }
        }
        @keyframes spin { 
            0% { transform: rotate(0deg); opacity: 0.6; } 
            50% { transform: rotate(180deg); opacity: 1; } 
            100% { transform: rotate(360deg); opacity: 0.6; } 
        }
    </style>
    """
    
    html_template = """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh; text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 20px; animation: spin 3s linear infinite;">📚</div>
        <h2 style="font-family: 'Songti SC', serif; color: #5d4037; font-weight: normal;">正在初始化知识库</h2>
        <p style="color: #8d6e63; font-style: italic; margin-top: 10px; font-size: 1.1rem;">{quote_text}</p>
        
        <div style="width: 300px; height: 4px; background: #efebe9; border-radius: 2px; margin-top: 30px; overflow: hidden;">
            <div style="width: 100%; height: 100%; background: linear-gradient(90deg, #8d6e63, #d7ccc8); animation: progress-indeterminate 1.5s infinite ease-in-out;"></div>
        </div>
        
        <p style="color: #bcaaa4; font-size: 0.8rem; margin-top: 15px;">首次加载可能需要几分钟，请耐心等待...</p>
    </div>
    """
    
    final_html = html_template.format(quote_text=quote)
    
    try:
        st.html(css_code + final_html)
    except AttributeError:
        st.markdown(css_code + final_html, unsafe_allow_html=True)

# ==========================================
# 3. 辅助与渲染函数 (Helpers & Renderers)
# ==========================================
def init_session_params():
    """初始化会话参数"""
    for param_key, config in PARAM_CONFIG.items():
        if param_key not in st.session_state:
            st.session_state[param_key] = config['default']
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def render_param_control(param_key, session_key, is_urgent, is_home):
    """渲染单个参数控件"""
    config = PARAM_CONFIG.get(param_key)
    if not config or not is_home: 
        return False
    
    current_val = st.session_state.get(param_key, config['default'])
    changed = False
    
    label_class = "param-label-mini urgent" if is_urgent else "param-label-mini"
    label_text = f"🔥 {config['label']}" if is_urgent else config['label']
    
    st.markdown(f'<div class="param-control-area"><span class="{label_class}">{label_text}</span></div>', unsafe_allow_html=True)
    new_val = current_val
    
    if config['type'] == 'select':
        options = config['options']
        fmt_func = lambda x: config['format_map'].get(x, x)
        try: idx = options.index(current_val)
        except ValueError: idx = 0
        new_val = st.selectbox("策略", options=options, format_func=fmt_func, index=idx, key=session_key, label_visibility="collapsed")
    elif config['type'] == 'slider':
        new_val = st.slider("值", config['min'], config['max'], current_val, step=config.get('step', 1), key=session_key, label_visibility="collapsed")
    
    if new_val != current_val:
        st.session_state[param_key] = new_val
        changed = True
    return changed

@st.fragment(run_every=0.1)
def run_streaming_generation():
    """处理流式生成逻辑"""
    if not st.session_state.get('is_processing'):
        return

    last_msg_idx = len(st.session_state.messages) - 1
    if last_msg_idx < 0 or st.session_state.messages[last_msg_idx]["role"] != "assistant":
        return
        
    last_msg = st.session_state.messages[last_msg_idx]
    
    if not st.session_state.is_processing and last_msg["content"]: 
        return

    user_question = ""
    for i in range(last_msg_idx, -1, -1):
        if st.session_state.messages[i]["role"] == "user":
            user_question = st.session_state.messages[i]["content"]
            break
    
    current_params = {k: st.session_state[k] for k in PARAM_CONFIG.keys()}
    
    if 'response_generator' not in st.session_state or st.session_state.response_generator is None:
        try:
            st.session_state.response_generator = st.session_state.rag_engine.query(user_question, **current_params)
            st.session_state.current_stream_buffer = ""
        except Exception as e:
            st.session_state.messages[last_msg_idx]["content"] = f"❌ 启动失败：{str(e)}"
            st.session_state.is_processing = False
            st.session_state.pop('response_generator', None)
            st.rerun()
            return

    generator = st.session_state.response_generator
    
    try:
        chunk = next(generator)
        
        if chunk:
            st.session_state.current_stream_buffer += chunk
            st.session_state.messages[last_msg_idx]["content"] = st.session_state.current_stream_buffer + "▌"
            return 
        else:
            return
            
    except StopIteration:
        final_text = st.session_state.current_stream_buffer
        st.session_state.messages[last_msg_idx]["content"] = final_text
        
        st.session_state.is_processing = False
        st.session_state.pop('response_generator', None)
        st.session_state.pop('current_stream_buffer', None)
        
        st.rerun()
        
    except Exception as e:
        error_msg = f"\n\n❌ 生成中断：{str(e)}"
        st.session_state.messages[last_msg_idx]["content"] = st.session_state.current_stream_buffer + error_msg
        st.session_state.is_processing = False
        st.session_state.pop('response_generator', None)
        st.session_state.pop('current_stream_buffer', None)
        st.rerun()

# ==========================================
# 4. 主程序入口 (Main Execution)
# ==========================================
def main():
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    init_session_params()
    
    st.markdown("""
    <div class='main-header'>
        <h1>📚 《遥远的救世主》读书小精灵</h1>
        <div class='subtitle'>悟道·强势文化 · 智能检索与推理助手</div>
    </div>
    """, unsafe_allow_html=True)

    engine_ready = 'rag_engine' in st.session_state
    
    if not engine_ready:
        if st.session_state.get('is_loading_engine', False):
            render_elegant_loading()
            if 'loading_error' not in st.session_state:
                try:
                    book_path = "./遥远的救世主.txt"
                    if not os.path.exists(book_path):
                        raise FileNotFoundError(f"未找到文件：{book_path}")
                    
                    engine_instance = RAGEngine(txt_path=book_path)
                    engine_instance.load_data() 
                    
                    st.session_state.rag_engine = engine_instance
                    st.session_state.is_loading_engine = False
                    st.rerun()
                except Exception as e:
                    st.session_state.is_loading_engine = False
                    st.session_state.loading_error = str(e)
                    st.rerun()
        elif st.session_state.get('loading_error'):
            st.markdown(f"""
            <div style="text-align: center; padding: 50px; color: #d84315;">
                <h3>❌ 初始化失败</h3>
                <p>{st.session_state.loading_error}</p>
                <button onclick="window.location.reload()" style="padding: 10px 20px; background: #5d4037; color: white; border: none; border-radius: 5px; cursor: pointer; margin-top:20px;">重新尝试</button>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 首次进入，开始加载
            st.session_state.is_loading_engine = True
            st.rerun()
    else:
        render_main_interface()

def render_main_interface():
    """渲染主界面逻辑"""
    engine = st.session_state.rag_engine
    metrics = getattr(engine, 'metrics', {})
    stages = metrics.get('stage_durations', {})
    debug_data = getattr(engine, 'last_retrieval_debug_info', [])
    bottleneck = max(stages, key=stages.get) if stages else None

    # ✅ 关键修改：使用标准的 st.columns，不再包裹任何强制高度的 div
    col_left, col_center, col_right = st.columns([1.2, 3.8, 1.2])

    # --- 左侧：技术栈 ---
    with col_left:
        # 不再需要 class='sidebar-panel' 来限制高度，但保留样式类用于美观
        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>⚙️ 技术架构</div>", unsafe_allow_html=True)
        for tech in TECH_STACK:
            st.markdown(f"""
            <div class="tech-item">
                <div class="tech-label"><span class="badge-tech">{tech['badge']}</span> {tech['label']}</div>
                <div class="tech-value">{tech['value']}</div>
                <div class="tech-desc">{tech['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 中间：核心交互区 ---
    with col_center:
        # 聊天历史与输入
        st.markdown("<div class='chat-section'>", unsafe_allow_html=True)
        st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if (st.session_state.is_processing and 
                        i == len(st.session_state.messages) - 1 and 
                        (not message["content"] or message["content"].strip() == "")):
                        render_zen_thinking()
                    elif (st.session_state.is_processing and 
                          i == len(st.session_state.messages) - 1 and 
                          message["content"]):
                        content = message["content"]
                        if not content.endswith("▌"):
                            display_content = content + "▌"
                        else:
                            display_content = content
                        st.markdown(display_content)
                    else:
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])
        
        if len(st.session_state.messages) == 0:
            st.markdown('<div style="text-align: center; color: #78909c; padding: 20px 0; font-style: italic;"><p>👋 您好！知识库已就绪。<br>请问关于《遥远的救世主》，您想了解什么？</p></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True) 
        
        st.markdown("<div class='chat-input-area'>", unsafe_allow_html=True)
        prompt = st.chat_input(
            "请输入问题...",
            key="unique_chat_input_main",
            disabled=(st.session_state.is_processing)
        )
        st.markdown("</div>", unsafe_allow_html=True) 
        st.markdown("</div>", unsafe_allow_html=True) 

        if prompt and not st.session_state.is_processing:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": ""})
            st.session_state.is_processing = True
            st.rerun()

        if st.session_state.is_processing:
            run_streaming_generation()

        # 底部控制台：链路监控
        with st.expander("🎛️ 链路监控与调优", expanded=True, key="monitor_expander"):
            has_data = bool(stages) and not st.session_state.is_processing
            
            if has_data:
                total_time = metrics.get('latency_seconds', 0)
                retry_count = metrics.get('attempts_made', 1)
                retry_info = f" (触发了 {retry_count-1} 次重试)" if retry_count > 1 else ""
                eval_time = stages.get('Eval_Phase', 0)
                fast_filter_hint = " ⚡ (Rerank 分数直判)" if eval_time < 0.1 and eval_time > 0 else ""
                st.caption(f"本轮耗时：**{total_time:.2f}s**{retry_info}{fast_filter_hint} | 瓶颈节点已高亮 (红色)")
            elif st.session_state.is_processing:
                st.caption("⏳ 正在计算链路耗时...")
            else:
                if len(st.session_state.messages) > 0:
                    st.caption("ℹ️ 本轮详细数据未捕获，以下为默认参数配置。")
                else:
                    st.caption("💡 提问后将在此处显示详细链路数据与调优参数。")
            
            cols_count = len(PIPELINE_NODES)
            
            html_container = '<div class="pipeline-container">'
            for i, node in enumerate(PIPELINE_NODES):
                is_bottleneck_node = False
                node_time = 0
                
                if stages:
                    matched_keys = [k for k in stages.keys() if node['stage_key'] in k]
                    if matched_keys:
                        node_time = sum(stages[k] for k in matched_keys)
                    
                    if bottleneck and bottleneck in node['stage_key']:
                        is_bottleneck_node = True
                    if bottleneck and 'Retry' in str(bottleneck) and node['id'] == 'eval':
                        is_bottleneck_node = True
                
                if st.session_state.is_processing:
                    time_str = "⏳"
                    time_class = "node-time"
                elif node_time > 0:
                    time_str = f"{node_time:.2f}s"
                    time_class = "node-time slow" if node_time > 0.8 else "node-time"
                else:
                    time_str = "--"
                    time_class = "node-time"
                
                card_class = "node-card bottleneck" if is_bottleneck_node else "node-card"
                
                html_container += f"""
                <div style="flex: 1; min-width: 0; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                    <div class="{card_class}" style="width: 100%; box-sizing: border-box;">
                        <span class="node-icon">{node['icon']}</span>
                        <div class="node-title">{node['title']}</div>
                        <div class="{time_class}">{time_str}</div>
                    </div>
                """
                if is_bottleneck_node:
                    tip_text = "🐢 瓶颈" if not (bottleneck and 'Retry' in str(bottleneck)) else "🔄 重试中"
                    html_container += f'<div class="optimization-tip" style="width:100%">{tip_text}</div>'
                html_container += "</div>"
            html_container += "</div>"
            st.markdown(html_container, unsafe_allow_html=True)
            
            control_cols = st.columns(cols_count, gap="small")
            for i, node in enumerate(PIPELINE_NODES):
                with control_cols[i]:
                    is_bottleneck_node = False
                    if bottleneck:
                        if bottleneck == node['stage_key']:
                            is_bottleneck_node = True
                        elif 'Retry' in str(bottleneck) and node['id'] == 'eval':
                            is_bottleneck_node = True
                    
                    has_home_control = False
                    for param in node['controlled_by']:
                        param_config = PARAM_CONFIG.get(param)
                        if not param_config: continue
                        
                        session_key = f"ctrl_{node['id']}_{param}" 
                        is_home = (param_config.get('home_node_id') == node['id'])
                        
                        if is_home:
                            has_home_control = True
                            if render_param_control(param, session_key, is_bottleneck_node, True):
                                st.rerun()
                    
                    if not has_home_control:
                        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                    for param in node['controlled_by']:
                        param_config = PARAM_CONFIG.get(param)
                        if not param_config: continue
                        is_home = (param_config.get('home_node_id') == node['id'])
                        if not is_home:
                            home_node_id = param_config.get('home_node_id')
                            home_node_title = next((n['title'] for n in PIPELINE_NODES if n['id'] == home_node_id), "未知节点")
                            ref_style = "color: #d84315; background: #fff3e0; border-color: #ffccbc;" if is_bottleneck_node else ""
                            st.markdown(f'<div class="param-ref-tag" style="{ref_style}">受 {home_node_title} 影响</div>', unsafe_allow_html=True)

        # ==========================================
        # 参数映射指南 (最终修复：无注释 + dedent)
        # ==========================================
        with st.expander("参数映射指南", expanded=False, key="guide_expander"):
            st.markdown("了解每个参数如何影响检索与生成的全过程。")
            
            # 关键点：
            # 1. 使用 textwrap.dedent 去除 Python 缩进带来的空格
            # 2. 三引号紧接 f"""，后面紧跟 <div，中间不要有空行
            # 3. 移除了所有 <!-- --> 注释，防止解析歧义
            guide_html = textwrap.dedent(f"""<div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px;">
                <div style="flex: 1; min-width: 250px; background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                    <div style="font-weight:bold; color:#5d4037; margin-bottom:10px; border-bottom:2px solid #d7ccc8; padding-bottom:5px; font-size:1.0rem;">📥 数据获取端</div>
                    <div style="margin-bottom:10px; padding:8px; background:#fff; border-radius:6px; border-left:3px solid #8d6e63;">
                        <div style="font-weight:bold; font-size:0.9rem; color:#3e2723;">{PARAM_CONFIG["top_k_initial"]["label"]}</div>
                        <div style="font-size:0.8rem; color:#546e7a; margin:2px 0;">{PARAM_CONFIG["top_k_initial"].get('desc_short', '')}</div>
                        <code style="background:#efebe9; color:#d84315; padding:2px 6px; border-radius:4px; font-size:0.75rem;">top_k_initial</code>
                    </div>
                    <div style="margin-bottom:10px; padding:8px; background:#fff; border-radius:6px; border-left:3px solid #8d6e63;">
                        <div style="font-weight:bold; font-size:0.9rem; color:#3e2723;">{PARAM_CONFIG["multi_query_count"]["label"]}</div>
                        <div style="font-size:0.8rem; color:#546e7a; margin:2px 0;">{PARAM_CONFIG["multi_query_count"].get('desc_short', '')}</div>
                        <code style="background:#efebe9; color:#d84315; padding:2px 6px; border-radius:4px; font-size:0.75rem;">multi_query_count</code>
                    </div>
                    <div style="border-top:1px dashed #cfd8dc; margin:10px 0;"></div>
                    <div style="font-weight:bold; color:#5d4037; margin-bottom:8px; font-size:0.9rem;">⚖️ 精细重排</div>
                    <div style="margin-bottom:10px; padding:8px; background:#fff; border-radius:6px; border-left:3px solid #8d6e63;">
                        <div style="font-weight:bold; font-size:0.9rem; color:#3e2723;">{PARAM_CONFIG["rerank_threshold"]["label"]}</div>
                        <div style="font-size:0.8rem; color:#546e7a; margin:2px 0;">{PARAM_CONFIG["rerank_threshold"].get('desc_short', '')}</div>
                        <code style="background:#efebe9; color:#d84315; padding:2px 6px; border-radius:4px; font-size:0.75rem;">rerank_threshold</code>
                    </div>
                    <div style="margin-bottom:10px; padding:8px; background:#fff; border-radius:6px; border-left:3px solid #8d6e63;">
                        <div style="font-weight:bold; font-size:0.9rem; color:#3e2723;">{PARAM_CONFIG["top_k_final"]["label"]}</div>
                        <div style="font-size:0.8rem; color:#546e7a; margin:2px 0;">{PARAM_CONFIG["top_k_final"].get('desc_short', '')}</div>
                        <code style="background:#efebe9; color:#d84315; padding:2px 6px; border-radius:4px; font-size:0.75rem;">top_k_final</code>
                    </div>
                </div>
                <div style="flex: 1; min-width: 250px; background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                    <div style="font-weight:bold; color:#5d4037; margin-bottom:10px; border-bottom:2px solid #d7ccc8; padding-bottom:5px; font-size:1.0rem;">📤 内容生产端</div>
                    <div style="margin-bottom:10px; padding:8px; background:#fff; border-radius:6px; border-left:3px solid #8d6e63;">
                        <div style="font-weight:bold; font-size:0.9rem; color:#3e2723;">{PARAM_CONFIG["rewrite_mode"]["label"]}</div>
                        <div style="font-size:0.8rem; color:#546e7a; margin:2px 0;">Query 重写模式</div>
                        <code style="background:#efebe9; color:#d84315; padding:2px 6px; border-radius:4px; font-size:0.75rem;">rewrite_mode</code>
                    </div>
                    <div style="border-top:1px dashed #cfd8dc; margin:10px 0;"></div>
                    <div style="margin-bottom:10px; padding:8px; background:#fff; border-radius:6px; border-left:3px solid #8d6e63;">
                        <div style="font-weight:bold; font-size:0.9rem; color:#3e2723;">{PARAM_CONFIG["max_self_rag_attempts"]["label"]}</div>
                        <div style="font-size:0.8rem; color:#546e7a; margin:2px 0;">自动重试次数 (智能降级)</div>
                        <code style="background:#efebe9; color:#d84315; padding:2px 6px; border-radius:4px; font-size:0.75rem;">max_self_rag_attempts</code>
                    </div>
                </div>
            </div>""")
            
            st.markdown(guide_html, unsafe_allow_html=True)

    # --- 右侧：证据溯源 ---
    with col_right:
        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        
        if st.session_state.is_processing:
            st.markdown('<div class="empty-state"><p style="font-size: 1.5rem;">🌀</p><p>正在深入思考</p><p style="font-size: 0.8rem; color: #90a4ae; margin-top:5px;">检索原文依据中...</p></div>', unsafe_allow_html=True)
        elif debug_data:
            st.markdown("<div class='panel-title'>📜 依据与溯源</div>", unsafe_allow_html=True)
            st.markdown("<div class='evidence-list'>", unsafe_allow_html=True)
            for item in debug_data[:8]:
                css_class = "evidence-card cited" if item.get("is_cited") else "evidence-card"
                badge = "<span class='badge-cited'>✓ 引用</span>" if item.get("is_cited") else "<span class='badge-uncited'>参考</span>"
                chapter = item.get('chapter', '未知')
                content = item.get('content', '')
                st.markdown(f"""
                <div class="{css_class}">
                    <div class="evidence-meta"><span>📖 {chapter}</span>{badge}</div>
                    <div class="evidence-text">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if len(debug_data) > 8:
                st.caption(f"💡 还有 {len(debug_data)-8} 个片段在后台参与计算。")
        else:
            st.markdown('<div class="empty-state"><p style="font-size: 1.5rem;">📚</p><p>暂无检索记录</p><p style="font-size: 0.8rem; color: #90a4ae; margin-top:5px;">提问后，此处将显示<br>原文依据与出处</p></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()