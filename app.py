import streamlit as st
from rag_engine import RAGEngine
import os
import json

# 页面配置
st.set_page_config(
    page_title="强势文化专家 - 遥远的救世主 RAG",
    page_icon="📜",
    layout="wide"
)

# 自定义 CSS (文雅范·水墨书香风格)
st.markdown("""
<style>
    /* --- 全局背景与字体：暖色调，仿纸质阅读体验 --- */
    .stApp {
        background-color: #fdfbf7; /* 暖羊皮纸色 */
    }
    .stChatMessage { 
        font-family: 'Georgia', 'Songti SC', 'SimSun', serif; /* 衬线体更有书卷气 */
        color: #2c3e50; /* 深炭灰，非纯黑 */
        line-height: 1.9; /* 增加行高，更透气 */
        background-color: transparent;
        border-bottom: 1px solid #eee;
        padding-bottom: 20px;
    }
    
    /* --- 标题层级：黛蓝与赭石，沉稳不刺眼 --- */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Songti SC', 'STSong', serif;
        font-weight: 600;
        color: #34495e; /* 黛蓝色 */
        margin-top: 1.2em;
        margin-bottom: 0.6em;
    }
    h3 { 
        border-bottom: 1px solid #d7ccc8; /* 浅褐色分割线 */
        padding-bottom: 0.3em;
        font-size: 1.35rem;
        color: #5d4037; /* 深赭石色 */
    }
    h4 { 
        color: #546e7a; /* 青灰色 */
        font-size: 1.1rem;
        font-style: italic; /* 稍微倾斜，增加文艺感 */
    }

    /* --- 重点文字高亮：仿古籍批注 --- */
    strong { 
        color: #5d4037; /* 深褐色文字 */
        background-color: #efebe9; /* 淡茶色背景 */
        padding: 0 3px; 
        border-radius: 2px; 
        font-weight: 700;
        border-bottom: 1px solid #d7ccc8; /* 下划线装饰 */
    }

    /* --- 列表样式：简洁圆点 --- */
    ul { 
        margin-bottom: 1em; 
        padding-left: 1.5em; 
    }
    li { 
        margin-bottom: 0.6em; 
        color: #455a64;
    }
    li::marker {
        color: #8d6e63; /* 褐色圆点 */
    }

    /* --- 引用块：静谧青灰，仿侧边批注 --- */
    blockquote { 
        border-left: 4px solid #b0bec5; /* 青灰色边框 */
        background-color: #f5f7f8; /* 极淡的青灰背景 */
        margin: 1.2em 0; 
        padding: 12px 16px; 
        color: #546e7a; 
        font-style: italic; 
        border-radius: 0 4px 4px 0; 
        font-family: 'KaiTi', '楷体', serif; /* 引用用楷体，区分正文 */
    }

    /* --- 动态呼吸点：改为淡雅的墨色 --- */
    @keyframes blink {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    .thinking-dots span {
        display: inline-block; width: 6px; height: 6px; margin: 0 4px;
        background-color: #546e7a; /* 青灰 */
        border-radius: 50%;
        animation: blink 1.5s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(1) { animation-delay: 0s; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    .thinking-container {
        display: flex; align-items: center; color: #546e7a; font-weight: 500;
        padding: 10px 14px; background-color: #eceff1; border-radius: 6px;
        border-left: 4px solid #90a4ae; margin-bottom: 15px;
        font-family: sans-serif; font-size: 0.9rem;
    }

    /* --- 状态提示框：柔和的警示色 --- */
    .crag-status-info {
        background-color: #fff8e1; border-left: 4px solid #ffe082;
        padding: 10px; margin-bottom: 10px; border-radius: 4px;
        color: #795548; font-family: sans-serif; font-size: 0.9rem;
        animation: fadeIn 0.3s;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }

    /* --- 证据链卡片：素雅风格 --- */
    .evidence-card {
        border-left: 3px solid #cfd8dc; background-color: #ffffff;
        padding: 12px; margin-bottom: 12px; border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace; font-size: 0.82rem; white-space: pre-wrap;
        line-height: 1.6; transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* 轻微阴影 */
        color: #607d8b;
    }
    .evidence-card:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 4px 8px rgba(0,0,0,0.08); 
        border-left-color: #78909c;
    }
    /* 引用过的卡片：墨绿色调 */
    .evidence-card.cited { 
        border-left-color: #66bb6a; 
        background-color: #f1f8e9; /* 极淡的绿意 */
        color: #33691e;
    }
    /* 未引用的卡片：保持素灰 */
    .evidence-card.ignored { 
        border-left-color: #b0bec5; 
        background-color: #fafafa; 
        opacity: 0.9; 
    }
    
    .evidence-header {
        display: flex; justify-content: space-between; align-items: center;
        font-size: 0.75rem; color: #78909c; margin-bottom: 8px;
        font-family: sans-serif; font-weight: 600; flex-wrap: wrap; gap: 5px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .chapter-badge { 
        background: #cfd8dc; color: #455a64; 
        padding: 2px 6px; border-radius: 3px; 
        font-size: 0.7em; font-weight: normal; 
    }
    .score-badge { 
        background: #eceff1; color: #546e7a; 
        padding: 2px 6px; border-radius: 3px; 
    }
    
    .rerank-badge { 
        background: #fff; color: #546e7a; padding: 2px 6px; 
        border-radius: 3px; font-size: 0.7em; margin-left: 5px; 
        border: 1px solid #cfd8dc; font-weight: bold;
    }
    .rerank-high { 
        background: #f1f8e9; color: #558b2f; border-color: #c5e1a5; 
    }

    /* --- 🆕 性能诊断面板样式 --- */
    .perf-card {
        background: #fdfbf7;
        border: 1px solid #d7ccc8;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
        transition: all 0.2s;
    }
    .perf-card:hover {
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border-color: #bcaaa4;
    }
    .perf-label {
        font-size: 0.75rem;
        color: #795548;
        font-family: 'Songti SC', serif;
        margin-bottom: 4px;
    }
    .perf-value {
        font-size: 1.3rem;
        font-weight: bold;
        font-family: 'Consolas', monospace;
        color: #5d4037;
    }
    .perf-value.bottleneck {
        color: #c62828; /* 赭红，标示瓶颈 */
    }
    .perf-bar-container {
        width: 100%;
        background-color: #eceff1;
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 4px;
    }
    .perf-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease-in-out;
    }
    .perf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        color: #546e7a;
        margin-bottom: 6px;
        font-family: 'Songti SC', serif;
    }
    .perf-time {
        font-family: 'Consolas', monospace;
        font-size: 0.8rem;
        color: #78909c;
    }

    /* 输入框美化 */
    .stChatInputContainer > div {
        background-color: #ffffff;
        border: 1px solid #d7ccc8;
        box-shadow: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("📜 强势文化专家系统")
st.markdown("基于《遥远的救世主》深度定制。**风格**：结构化排版 · 水墨书香 · 深度推演")

# 侧边栏
with st.sidebar:
    st.header("控制面板")
    st.markdown("---")
    book_path = "./遥远的救世主.txt"
    
    if not os.path.exists(book_path):
        st.error(f"未找到文件：{book_path}")
        st.stop()
    
    if 'rag_engine' not in st.session_state:
        with st.spinner("正在加载知识库与模型..."):
            try:
                engine_instance = RAGEngine(txt_path=book_path)
                engine_instance.load_data() 
                st.session_state.rag_engine = engine_instance
                st.success("✅ 系统就绪")
            except Exception as e:
                st.error(f"初始化失败：{str(e)}")
                st.stop()
    
    engine = st.session_state.rag_engine

    st.subheader("知识库")
    if st.button("🔄 更新索引", use_container_width=True):
        with st.spinner("处理中..."):
            try:
                engine._data_loaded_attempted = False 
                engine.load_data()
                st.success("索引已更新")
                st.rerun()
            except Exception as e:
                st.error(f"错误：{str(e)}")
    
    db_path = "./chroma_db"
    is_ready = os.path.exists(db_path) and any(os.scandir(db_path))

    if is_ready:
        st.success("✅ 知识库在线")
        # 进度显示
        if os.path.exists("./checkpoint.json") and engine.full_text:
            try:
                with open("./checkpoint.json", 'r', encoding='utf-8') as f:
                    cp = json.load(f)
                    processed = cp.get('processed_chars', 0)
                    total = len(engine.full_text)
                    if total > 0:
                        st.caption(f"进度：{processed}/{total} ({(processed/total*100):.1f}%)")
            except Exception:
                pass
        
        st.markdown("---")
        eval_model_name = "未知"
        if hasattr(engine, 'eval_llm') and engine.eval_llm:
            if hasattr(engine.eval_llm, 'model'): eval_model_name = engine.eval_llm.model
            elif hasattr(engine.eval_llm, 'model_name'): eval_model_name = engine.eval_llm.model_name
            
        rerank_status = "BGE-Large" if hasattr(engine, 'reranker') and engine.reranker else "未加载"
        
        st.info(f"**模型配置**\n"
                f"- 生成：`{engine.model_name}`\n"
                f"- 评估：`{eval_model_name}`\n"
                f"- 重排：`{rerank_status}`", icon="ℹ️")
    else:
        st.warning("⚠️ 知识库未就绪")

    st.markdown("---")
    st.caption("💡 **提示**：系统采用**文雅模式**排版，适合深度阅读与思考。")

# 主聊天区域
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请问关于天道、文化属性或丁元英的任何问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        message_placeholder = st.empty()
        
        thinking_html = """
        <div class="thinking-container">
            <span>🧠 正在引经据典，深度推演...</span>
            <div class="thinking-dots" style="margin-left: 10px;">
                <span></span><span></span><span></span>
            </div>
        </div>
        """
        status_placeholder.markdown(thinking_html, unsafe_allow_html=True)
        
        full_response = ""
        has_started_typing = False
        
        try:
            response_stream = engine.query(prompt)
            
            for chunk in response_stream:
                if chunk:
                    if chunk.startswith("⚠️") or chunk.startswith("❌"):
                        status_placeholder.markdown(f"<div class='crag-status-info'>{chunk}</div>", unsafe_allow_html=True)
                        continue
                    
                    if chunk.startswith("💡"):
                        status_placeholder.markdown(
                            "<div class='crag-status-info' style='background-color:#f1f8e9; border-color:#a5d6a7; color:#2e7d32;'>✅ 检索完成，正在撰写分析报告...</div>", 
                            unsafe_allow_html=True
                        )
                        continue 
                    
                    if not has_started_typing:
                        status_placeholder.empty()
                        has_started_typing = True
                    
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            status_placeholder.empty()
            if full_response:
                message_placeholder.markdown(f"<div class='expert-response'>{full_response}</div>", unsafe_allow_html=True)
            else:
                message_placeholder.markdown("⚠️ 未能生成有效回答。")
            
            # === 证据链可视化 & 🆕 性能诊断 ===
            debug_data = getattr(engine, 'last_retrieval_debug_info', [])
            
            if debug_data:
                with st.expander("🔍 查阅：引用来源与诊断", expanded=False):
                    
                    # 🆕 1. 性能诊断面板 (如果有日志数据)
                    log_file = getattr(engine, 'log_file', None)
                    metrics = {}
                    if log_file and os.path.exists(log_file):
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                if lines:
                                    last_log = json.loads(lines[-1])
                                    metrics = last_log.get('metrics', {})
                        except Exception:
                            pass

                    if metrics.get('stage_durations'):
                        stages = metrics['stage_durations']
                        total_time = metrics.get('latency_seconds', 0)
                        risks = metrics.get('risk_flags', {})
                        
                        st.subheader("📊 推演过程 · 性能诊断")
                        st.caption("记录本次问答各环节耗时，助您洞察系统瓶颈。")
                        
                        # 找出瓶颈
                        bottleneck_stage = max(stages, key=stages.get) if stages else ""
                        
                        # 映射中文名称
                        stage_names = {
                            "Rewrite": "语义重写", "MultiQuery": "多路查询", 
                            "Retrieval": "并行检索", "Rerank": "重排序", 
                            "Eval": "质量评估", "Generation": "文章生成"
                        }
                        
                        # 颜色映射 (文雅色系)
                        colors = {
                            "Rewrite": "#5d4037", "MultiQuery": "#6d4c41", 
                            "Retrieval": "#546e7a", "Rerank": "#00897b", 
                            "Eval": "#7cb342", "Generation": "#34495e"
                        }

                        # 四大指标卡片
                        c1, c2, c3, c4 = st.columns(4)
                        
                        c1.markdown(f"""
                        <div class="perf-card">
                            <div class="perf-label">总耗时</div>
                            <div class="perf-value">{total_time:.2f}s</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        bn_name = next((v for k, v in stage_names.items() if k in bottleneck_stage), bottleneck_stage.split('_')[-1])
                        c2.markdown(f"""
                        <div class="perf-card">
                            <div class="perf-label">耗时瓶颈</div>
                            <div class="perf-value bottleneck">{bn_name}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c3.markdown(f"""
                        <div class="perf-card">
                            <div class="perf-label">检索片段</div>
                            <div class="perf-value">{metrics.get('total_retrieved', 0)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        rate = metrics.get('citation_rate', 0)
                        c4.markdown(f"""
                        <div class="perf-card">
                            <div class="perf-label">引用率</div>
                            <div class="perf-value" style="color: {'#2e7d32' if rate > 0.5 else '#f57f17'}">{rate:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                        
                        # 耗时分布条形图
                        st.markdown("#### ⏳ 环节耗时分布")
                        sorted_stages = sorted([(k, v) for k, v in stages.items() if v > 0.01], key=lambda x: x[1], reverse=True)
                        
                        for key, duration in sorted_stages:
                            # 匹配颜色
                            color = next((v for k, v in colors.items() if k in key), "#90a4ae")
                            width_pct = (duration / total_time) * 100 if total_time > 0 else 0
                            label = next((v for k, v in stage_names.items() if k in key), key.split('_')[-1])
                            
                            st.markdown(f"""
                            <div class="perf-row">
                                <span>{label}</span>
                                <span class="perf-time">{duration:.3f}s ({width_pct:.1f}%)</span>
                            </div>
                            <div class="perf-bar-container">
                                <div class="perf-bar-fill" style="width: {width_pct}%; background-color: {color};"></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 风险警示
                        active_risks = [k for k, v in risks.items() if v]
                        if active_risks:
                            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                            risk_msgs = {
                                "hallucination_risk": "⚠️ **幻觉风险**：检索内容未被充分引用。",
                                "low_utilization": "ℹ️ **利用率低**：检索过多但引用过少。",
                                "high_latency": "⏳ **响应缓慢**：总耗时较长。"
                            }
                            for r in active_risks:
                                msg = risk_msgs.get(r, r)
                                st.markdown(f"<div style='background:#fff3e0; border-left:4px solid #ffb74d; padding:8px; border-radius:4px; color:#e65100; font-size:0.9rem; margin-bottom:5px;'>{msg}</div>", unsafe_allow_html=True)
                        
                        st.markdown("---")

                    # 2. 原有的证据链统计
                    st.caption("💡 **墨绿卡片**：已被引用。 **素灰卡片**：参考但未引用。所有片段经 **BGE Rerank** 重排序。")
                    
                    total_docs = len(debug_data)
                    cited_docs = sum(1 for x in debug_data if x.get("is_cited", False))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("检索片段", total_docs)
                    c2.metric("实际引用", cited_docs)
                    c3.metric("利用率", f"{cited_docs/total_docs:.1%}" if total_docs > 0 else "0%")
                    
                    for item in debug_data:
                        is_cited = item.get("is_cited", False)
                        score = item.get("score", 0)
                        content = item.get("content", "")
                        idx = item.get("index", 0)
                        chapter = item.get("chapter", "未知")
                        rerank_score = item.get("rerank_score", 0) 
                        
                        card_class = "cited" if is_cited else "ignored"
                        status_icon = "✓" if is_cited else "○"
                        
                        if rerank_score > 0.8:
                            rerank_label = f"极高 ({rerank_score:.2f})"
                            rerank_class = "rerank-high"
                        elif rerank_score > 0.5:
                            rerank_label = f"高 ({rerank_score:.2f})"
                            rerank_class = ""
                        else:
                            rerank_label = f"参考 ({rerank_score:.2f})"
                            rerank_class = ""
                        
                        preview_content = content[:200] + "..." if len(content) > 200 else content
                        
                        html_card = f"""
                        <div class="evidence-card {card_class}">
                            <div class="evidence-header">
                                <div>
                                    <span>{status_icon} 片段 #{idx}</span>
                                    <span class="chapter-badge">{chapter}</span>
                                    <span class="rerank-badge {rerank_class}">{rerank_label}</span>
                                </div>
                                <span class="score-badge">相似度：{score:.3f}</span>
                            </div>
                            <div>{preview_content}</div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                    
                    # 诊断逻辑
                    if total_docs == 0:
                        st.error("❌ **检索失败**：未找到相关片段。")
                    elif cited_docs == 0:
                        high_score_uncited = any(x.get('rerank_score', 0) > 0.8 and not x.get('is_cited') for x in debug_data)
                        if high_score_uncited:
                            st.warning("⚠️ **生成保守**：有高相关片段但未引用。")
                        else:
                            st.warning("⚠️ **生成失效**：未引用任何内容。")
                    elif cited_docs < total_docs * 0.3:
                        st.info(f"ℹ️ **精准模式**：引用了 {cited_docs} 个核心片段。")
                    else:
                        st.success("✅ **综合模式**：充分利用了多个片段。")

        except Exception as e:
            status_placeholder.empty()
            st.error(f"❌ **系统错误**: {str(e)}")
            full_response = f"❌ **系统错误**: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": full_response})