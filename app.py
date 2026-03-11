import streamlit as st
from rag_engine import RAGEngine
import os
import time

# ==========================================
# 1. 配置与常量区 (Configuration & Constants)
# ==========================================
PAGE_CONFIG = {
    "page_title": "《遥远的救世主》读书小精灵",
    "page_icon": "📚",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

# ⚠️ 调整：更新描述以反映后端的智能优化
PARAM_CONFIG = {
    "rewrite_mode": {"label": "改写策略", "type": "select", "options": ["direct", "light", "deep"], "format_map": {"direct": "直连", "light": "轻量", "deep": "深度"}, "default": "deep", "desc_short": "Query 重写模式", "home_node_id": "rewrite"},
    "multi_query_count": {"label": "发散路数", "type": "slider", "min": 0, "max": 5, "step": 1, "default": 1, "desc_short": "多路查询变体数量", "home_node_id": "multi_query"},
    # 更新描述：提示有快速过滤机制
    "top_k_initial": {"label": "初召 TopK", "type": "slider", "min": 5, "max": 50, "step": 5, "default": 8, "desc_short": "混合召回数量 (含关键词预检加速)", "home_node_id": "retrieval"},
    "rerank_threshold": {"label": "重排门槛", "type": "slider", "min": 0.0, "max": 0.9, "step": 0.05, "default": 0.5, "desc_short": "低于此分数的片段将被丢弃", "home_node_id": "rerank"},
    # 更新描述：提示 CoT 和智能降级
    "max_self_rag_attempts": {"label": "最大重试", "type": "slider", "min": 0, "max": 3, "step": 1, "default": 1, "desc_short": "失败重试 (含 CoT 评估 + 智能降级)", "home_node_id": "eval"},
    "top_k_final": {"label": "终选 TopK", "type": "slider", "min": 1, "max": 10, "step": 1, "default": 3, "desc_short": "最终送入上下文的最大片段数", "home_node_id": "generation"}
}

# ⚠️ 调整：stage_key 必须与后端 timer.checkpoint() 的字符串精确匹配
PIPELINE_NODES = [
    {"id": "rewrite", "title": "语义重写", "icon": "🧠", "controlled_by": ["rewrite_mode"], "desc": "Query 转化", "stage_key": "Query_Rewrite"},
    {"id": "multi_query", "title": "多路发散", "icon": "🔀", "controlled_by": ["multi_query_count", "rewrite_mode"], "desc": "多角度变体", "stage_key": "MultiQuery_Gen"},
    {"id": "retrieval", "title": "混合检索", "icon": "📚", "controlled_by": ["top_k_initial"], "desc": "向量 + 关键词", "stage_key": "Parallel_Retrieval"},
    {"id": "rerank", "title": "重排序", "icon": "⚖️", "controlled_by": ["top_k_initial", "rerank_threshold", "top_k_final"], "desc": "精细打分", "stage_key": "Rerank"},
    # 保留 Eval 节点以展示 Self-RAG 逻辑，尽管后端将其合并在了检索循环中
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
# 2. 样式定义 (Styles) - 保持原样
# ==========================================
def get_custom_css():
    return """
    <style>
        .stApp { background-color: #fcfbf9; background-image: linear-gradient(to bottom, #fcfbf9 0%, #f7f5f0 100%); overflow: hidden !important; height: 100vh; display: flex; flex-direction: column; }
        body { font-family: 'Songti SC', 'STSong', 'Noto Serif SC', 'SimSun', serif; color: #2c2c2c; }
        .block-container { padding-top: 0.5rem !important; padding-bottom: 0rem !important; padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important; height: 100%; display: flex; flex-direction: column; gap: 10px; }
        .sidebar-panel { background: #fff; border-radius: 12px; border: 1px solid #efebe9; box-shadow: 0 4px 12px rgba(0,0,0,0.05); padding: 6px 6px; height: 100%; overflow-y: auto; display: flex; flex-direction: column; }
        .panel-title { font-size: 1.1rem; font-weight: bold; color: #5d4037; margin-bottom: 15px; border-left: 4px solid #8d6e63; padding-left: 10px; display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
        .tech-item { margin-bottom: 8px; padding: 6px; background: #f9f9f9; border-radius: 8px; border: 1px solid #eee; }
        .tech-label { font-size: 0.75rem; color: #90a4ae; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
        .tech-value { font-size: 0.95rem; color: #2c3e50; font-weight: bold; font-family: 'Consolas', monospace; }
        .tech-desc { font-size: 0.8rem; color: #546e7a; margin-top: 4px; line-height: 1.4; }
        .badge-tech { display: inline-block; background: #e3f2fd; color: #1976d2; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-right: 5px; }
        .evidence-list { display: flex; flex-direction: column; gap: 12px; }
        .evidence-card { padding: 12px; border-radius: 8px; background: #fafafa; border: 1px solid #eee; transition: all 0.3s; }
        .evidence-card.cited { background: #fff8e1; border-color: #ffcc80; box-shadow: 0 2px 6px rgba(255, 179, 0, 0.1); transform: translateX(2px); }
        .evidence-card:not(.cited) { opacity: 0.75; filter: grayscale(0.2); }
        .evidence-meta { font-size: 0.75rem; color: #8d6e63; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }
        .evidence-text { font-size: 0.85rem; color: #37474f; line-height: 1.6; font-family: 'Songti SC', serif; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .badge-cited { background: #ffb74d; color: #fff; padding: 1px 5px; border-radius: 3px; font-size: 0.65rem; font-weight: bold; }
        .badge-uncited { background: #cfd8dc; color: #fff; padding: 1px 5px; border-radius: 3px; font-size: 0.65rem; }
        .empty-state { text-align: center; color: #90a4ae; font-style: italic; padding: 20px 10px; font-size: 0.9rem; flex-grow: 1; display: flex; flex-direction: column; justify-content: center; }
        .center-column-wrapper { display: flex; flex-direction: column; height: 100%; gap: 10px; overflow: hidden; min-height: 0; }
        .main-header { text-align: center; padding: 5px 0 10px 0; flex-shrink: 0; }
        .main-header h1 { font-family: 'Songti SC', serif; font-weight: bold; color: #2c3e50; letter-spacing: 2px; margin-bottom: 2px !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); font-size: 1.8rem; }
        .subtitle { color: #78909c; font-size: 1rem; font-style: italic; }
        .chat-section { background: #fff; border-radius: 12px; border: 1px solid #e0e0e0; display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.03); overflow: hidden; }
        .chat-history { flex: 1 1 auto; overflow-y: auto; padding: 15px; scroll-behavior: smooth; min-height: 0; }
        .chat-input-area { padding: 10px 15px; border-top: 1px solid #eee; background: #fcfbf9; z-index: 100; flex-shrink: 0; }
        .stChatMessage { padding: 8px 0; border-bottom: 1px solid #f5f5f5; }
        .stChatMessage:last-child { border-bottom: none; }
        .bottom-console { background: #fff; border-radius: 12px; border: 1px solid #e0e0e0; padding: 10px 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); flex-shrink: 0; max-height: 50vh; overflow-y: auto; display: flex; flex-direction: column; gap: 5px; }
        .pipeline-container { display: flex; flex-wrap: nowrap; gap: 8px; justify-content: stretch; margin-bottom: 0px; padding-bottom: 0px; border-bottom: none; width: 100%; }
        .stColumn { display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: flex-start !important; padding: 0 2px !important; min-width: 0 !important; flex: unset !important; }
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
        .stSlider label, .stSelectbox label, .stNumberInput label { display: none !important; }
        .stSlider, .stSelectbox, .stNumberInput { margin-bottom: 0 !important; margin-top: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important; min-height: auto !important; width: 100% !important; }
        .stSlider .stTextInput input, .stSelectbox .stTextInput input { font-size: 0.5rem !important; height: 15px !important; padding: 1px 3px !important; }
        [data-baseweb="select"] span { font-size: 0.5rem !important; }
        div[data-baseweb="popover"] ul { font-size: 0.5rem !important; }
        div[data-baseweb="popover"] li { font-size: 0.5rem !important; padding: 4px 8px !important; }
        div[data-baseweb="popover"] li:hover { background: #f5f5f5 !important; }
        .stSlider div[data-baseweb="slider"] { margin-top: 2px !important; margin-bottom: 2px !important; }
        .loading-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #78909c; font-style: italic; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #8d6e63; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; margin-bottom: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        footer, header { visibility: hidden; height: 0; }
    </style>
    """

# ==========================================
# 3. 辅助与渲染函数 (Helpers & Renderers)
# ==========================================
def ensure_engine_loaded():
    """确保 RAG 引擎已加载到 Session State"""
    if 'rag_engine' in st.session_state:
        return True
    
    if 'is_loading_engine' not in st.session_state:
        st.session_state.is_loading_engine = True
        try:
            book_path = "./遥远的救世主.txt"
            if not os.path.exists(book_path):
                st.error(f"未找到文件：{book_path}")
                st.session_state.is_loading_engine = False
                return False
            
            engine_instance = RAGEngine(txt_path=book_path)
            engine_instance.load_data() 
            st.session_state.rag_engine = engine_instance
            st.session_state.is_loading_engine = False
            return True
        except Exception as e:
            st.session_state.loading_error = str(e)
            st.session_state.is_loading_engine = False
            return False
    return False

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

@st.fragment(run_every=0.5)
def run_streaming_generation():
    """处理流式生成逻辑"""
    if not st.session_state.get('is_processing'):
        return

    last_msg = st.session_state.messages[-1]
    # 如果已有完整回复且不在加载中，跳过
    if last_msg["role"] == "assistant" and last_msg["content"] != "" and not last_msg["content"].endswith("▌"): 
        return

    user_question = st.session_state.messages[-2]["content"]
    current_params = {k: st.session_state[k] for k in PARAM_CONFIG.keys()}
    last_msg_idx = len(st.session_state.messages) - 1
    
    # 初始化生成器
    if 'current_stream_buffer' not in st.session_state:
        st.session_state.current_stream_buffer = ""
        try:
            st.session_state.response_generator = st.session_state.rag_engine.query(user_question, **current_params)
        except Exception as e:
            st.session_state.messages[last_msg_idx]["content"] = f"❌ 发生错误：{str(e)}"
            st.session_state.is_processing = False
            st.session_state.pop('response_generator', None)
            st.session_state.pop('current_stream_buffer', None)
            return

    generator = st.session_state.get('response_generator')
    if not generator: return

    try:
        chunk = next(generator)
        if chunk:
            # 过滤非内容标记
            if not chunk.startswith(("⚠️", "❌", "💡")):
                st.session_state.current_stream_buffer += chunk
                st.session_state.messages[last_msg_idx]["content"] = st.session_state.current_stream_buffer + "▌"
            return 
        else:
            raise StopIteration
    except StopIteration:
        final_text = st.session_state.current_stream_buffer
        st.session_state.messages[last_msg_idx]["content"] = final_text
        st.session_state.is_processing = False
        st.session_state.pop('response_generator', None)
        st.session_state.pop('current_stream_buffer', None)
        st.rerun()
    except Exception as e:
        st.session_state.messages[last_msg_idx]["content"] = f"❌ 发生错误：{str(e)}"
        st.session_state.is_processing = False
        st.session_state.pop('response_generator', None)
        st.session_state.pop('current_stream_buffer', None)
        st.rerun()

# ==========================================
# 4. 主程序入口 (Main Execution)
# ==========================================
def main():
    # 页面配置
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # 初始化
    init_session_params()
    engine_ready = ensure_engine_loaded()

    # 获取监控数据
    metrics = {}
    stages = {}
    bottleneck = None
    debug_data = []

    if engine_ready and not st.session_state.is_processing:
        engine = st.session_state.rag_engine
        metrics = getattr(engine, 'metrics', {})
        stages = metrics.get('stage_durations', {})
        debug_data = getattr(engine, 'last_retrieval_debug_info', [])
        
        # 寻找耗时最长的节点作为瓶颈
        if stages:
            bottleneck = max(stages, key=stages.get)
        else:
            bottleneck = None

    # 布局：三列
    col_left, col_center, col_right = st.columns([1.2, 3.8, 1.2])

    # --- 左侧：技术栈 ---
    with col_left:
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
        st.markdown("<div class='center-column-wrapper'>", unsafe_allow_html=True)
        
        # 1. 标题
        st.markdown("""
        <div class='main-header'>
            <h1>📚 《遥远的救世主》读书小精灵</h1>
            <div class='subtitle'>悟道·强势文化 · 智能检索与推理助手</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. 聊天历史与输入
        st.markdown("<div class='chat-section'>", unsafe_allow_html=True)
        st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
        
        if not engine_ready:
            if st.session_state.get('is_loading_engine'):
                st.markdown('<div class="loading-state"><div class="spinner"></div><p>正在加载知识库与大模型...</p></div>', unsafe_allow_html=True)
                time.sleep(0.5)
                st.rerun()
            elif st.session_state.get('loading_error'):
                st.error(f"加载失败：{st.session_state.loading_error}")
            else:
                st.rerun()
        else:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    if (message["role"] == "assistant" and st.session_state.is_processing and 
                        i == len(st.session_state.messages) - 1):
                        content = message["content"]
                        if content == "":
                            st.markdown("**🧠 正在推演强势文化逻辑...**\n\n请稍候...▌")
                        else:
                            st.markdown(content)
                    else:
                        st.markdown(message["content"])
            
            if len(st.session_state.messages) == 0:
                st.markdown('<div style="text-align: center; color: #78909c; padding: 20px 0; font-style: italic;"><p>👋 您好！知识库已就绪。<br>请问关于《遥远的救世主》，您想了解什么？</p></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True) 
        
        st.markdown("<div class='chat-input-area'>", unsafe_allow_html=True)
        prompt = st.chat_input(
            "请输入问题...",
            key="unique_chat_input_main",
            disabled=(not engine_ready or st.session_state.is_processing)
        )
        st.markdown("</div>", unsafe_allow_html=True) 
        st.markdown("</div>", unsafe_allow_html=True) 

        # 处理用户输入
        if prompt and engine_ready and not st.session_state.is_processing:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": ""})
            st.session_state.is_processing = True
            st.rerun()

        if st.session_state.is_processing and engine_ready:
            run_streaming_generation()

        # 3. 底部控制台：链路监控
        st.markdown("<div class='bottom-console'>", unsafe_allow_html=True)

        if not engine_ready:
            st.markdown("<div style='text-align:center; color:#90a4ae; padding:20px;'>⏳ 系统初始化中...</div>")
        else:
            with st.expander("🎛️ 链路监控与调优", expanded=True):
                if not st.session_state.is_processing and stages:
                    total_time = metrics.get('latency_seconds', 0)
                    retry_count = metrics.get('attempts_made', 1)
                    retry_info = f" (触发了 {retry_count-1} 次重试)" if retry_count > 1 else ""
                    
                    # 检查是否触发了快速过滤 (通过比较 Eval_Phase 耗时，如果极短说明走了快速通道)
                    eval_time = stages.get('Eval_Phase', 0)
                    fast_filter_hint = " ⚡ (触发关键词预检加速)" if eval_time < 0.1 and eval_time > 0 else ""
                    st.caption(f"本轮耗时：**{total_time:.2f}s**{retry_info}{fast_filter_hint} | 瓶颈节点已高亮 (红色)")
                
                cols_count = len(PIPELINE_NODES)
                
                # 渲染视觉节点行 (HTML)
                html_container = '<div class="pipeline-container">'
                for i, node in enumerate(PIPELINE_NODES):
                    is_bottleneck_node = False
                    node_time = 0
                    
                    if not st.session_state.is_processing and stages:
                        # 模糊匹配 stage_key，防止后端 key 有细微变化导致显示为 0
                        matched_keys = [k for k in stages.keys() if node['stage_key'] in k]
                        if matched_keys:
                            node_time = sum(stages[k] for k in matched_keys)
                        
                        if bottleneck and bottleneck in node['stage_key']:
                            is_bottleneck_node = True
                        # 特殊处理：如果瓶颈是 Retry_Overhead，高亮 Eval 节点
                        if 'Retry' in bottleneck and node['id'] == 'eval':
                            is_bottleneck_node = True
                    
                    time_str = f"{node_time:.2f}s" if node_time > 0 else ("⏳" if st.session_state.is_processing else "--")
                    time_class = "node-time slow" if (node_time > 0.8 and not st.session_state.is_processing) else "node-time"
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
                        tip_text = "🐢 瓶颈" if 'Retry' not in bottleneck else "🔄 重试中"
                        html_container += f'<div class="optimization-tip" style="width:100%">{tip_text}</div>'
                    html_container += "</div>"
                html_container += "</div>"
                st.markdown(html_container, unsafe_allow_html=True)
                
                # 渲染控件行
                control_cols = st.columns(cols_count, gap="small")
                for i, node in enumerate(PIPELINE_NODES):
                    with control_cols[i]:
                        node_stage = node['stage_key']
                        is_bottleneck_node = (bottleneck == node_stage or (bottleneck and 'Retry' in bottleneck and node['id'] == 'eval')) if not st.session_state.is_processing and stages else False
                        
                        has_home_control = False
                        # 渲染主控件
                        for param in node['controlled_by']:
                            param_config = PARAM_CONFIG.get(param)
                            if not param_config: continue
                            session_key = f"node_{i}_{param}"
                            is_home = (param_config.get('home_node_id') == node['id'])
                            
                            if is_home:
                                has_home_control = True
                                if render_param_control(param, session_key, is_bottleneck_node, True):
                                    st.rerun()
                        
                        if not has_home_control:
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                        # 渲染引用提示
                        for param in node['controlled_by']:
                            param_config = PARAM_CONFIG.get(param)
                            if not param_config: continue
                            is_home = (param_config.get('home_node_id') == node['id'])
                            if not is_home:
                                home_node_id = param_config.get('home_node_id')
                                home_node_title = next((n['title'] for n in PIPELINE_NODES if n['id'] == home_node_id), "未知节点")
                                ref_style = "color: #d84315; background: #fff3e0; border-color: #ffccbc;" if is_bottleneck_node else ""
                                st.markdown(f'<div class="param-ref-tag" style="{ref_style}">受 {home_node_title} 影响</div>', unsafe_allow_html=True)

            # 参数映射指南
            with st.expander("参数映射指南", expanded=False):
                st.markdown("了解每个参数如何影响检索与生成的全过程。")
                c_in, c_out = st.columns(2)
                
                def render_compact_row(label, desc, code):
                    st.markdown(f"""
                    <div class="param-guide-row">
                        <div><span class="param-guide-name">{label}</span><span class="param-guide-desc">({desc})</span></div>
                        <div class="param-guide-code">{code}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c_in:
                    st.markdown("#### 📥 数据获取端")
                    render_compact_row(PARAM_CONFIG["top_k_initial"]["label"], PARAM_CONFIG["top_k_initial"].get('desc_short', ''), "top_k_initial")
                    render_compact_row(PARAM_CONFIG["multi_query_count"]["label"], PARAM_CONFIG["multi_query_count"].get('desc_short', ''), "multi_query_count")
                    st.markdown("---")
                    st.markdown("**⚖️ 精细重排**")
                    render_compact_row(PARAM_CONFIG["rerank_threshold"]["label"], PARAM_CONFIG["rerank_threshold"].get('desc_short', ''), "rerank_threshold")
                    render_compact_row(PARAM_CONFIG["top_k_final"]["label"], PARAM_CONFIG["top_k_final"].get('desc_short', ''), "top_k_final")
                    
                with c_out:
                    st.markdown("#### 📤 内容生产端")
                    render_compact_row(PARAM_CONFIG["rewrite_mode"]["label"], "Query 重写模式", "rewrite_mode")
                    st.markdown("---")
                    render_compact_row(PARAM_CONFIG["max_self_rag_attempts"]["label"], "自动重试次数 (智能降级)", "max_self_rag_attempts")
    
        st.markdown("</div>", unsafe_allow_html=True) 
        st.markdown("</div>", unsafe_allow_html=True) 

    # --- 右侧：证据溯源 ---
    with col_right:
        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        
        if not engine_ready:
            st.markdown('<div class="empty-state"><p style="font-size: 1.5rem;">⏳</p><p>系统初始化中</p></div>', unsafe_allow_html=True)
        elif st.session_state.is_processing:
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