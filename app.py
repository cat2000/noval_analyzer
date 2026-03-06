import streamlit as st
from rag_engine import RAGEngine
import os
import re

# 页面配置
st.set_page_config(
    page_title="强势文化专家 - 遥远的救世主 RAG",
    page_icon="📚",
    layout="wide"
)

# 自定义 CSS 美化 (包含动态呼吸灯动画 + 证据链样式)
st.markdown("""
<style>
    .stChatMessage {
        font-family: 'Georgia', serif;
    }
    .expert-response {
        border-left: 4px solid #d9534f;
        padding-left: 10px;
        background-color: #f9f9f9;
    }
    
    /* --- 动态呼吸点动画 --- */
    @keyframes blink {
        0%, 100% { opacity: 0.2; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    .thinking-dots span {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin: 0 3px;
        background-color: #0068c9;
        border-radius: 50%;
        animation: blink 1.4s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(1) { animation-delay: 0s; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    .thinking-container {
        display: flex;
        align-items: center;
        color: #0068c9;
        font-weight: 600;
        padding: 12px 16px;
        background-color: #e8f4fd;
        border-radius: 8px;
        border-left: 5px solid #0068c9;
        margin-bottom: 10px;
        font-family: sans-serif;
        font-size: 0.95rem;
    }

    /* --- 新增：证据链卡片样式 --- */
    .evidence-card {
        border-left: 5px solid #ccc;
        background-color: #f8f9fa;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        line-height: 1.4;
    }
    .evidence-card.cited {
        border-left-color: #2ecc71; /* 绿色：被引用 */
        background-color: #f0fff4;
    }
    .evidence-card.ignored {
        border-left-color: #bdc3c7; /* 灰色：未引用 */
        background-color: #f9f9f9;
        opacity: 0.8;
    }
    .evidence-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 6px;
        font-family: sans-serif;
        font-weight: bold;
    }
    .score-badge {
        background: #eee;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 强势文化专家系统")
st.markdown("基于《遥远的救世主》深度定制的 RAG 助手，本地运行 Ollama + Qwen。")

# 侧边栏：控制与状态
with st.sidebar:
    st.header("控制面板")
    
    book_path = "./遥远的救世主.txt"
    
    if not os.path.exists(book_path):
        st.error(f"未找到文件：{book_path}，请将小说 txt 放在同目录下。")
        st.stop()
    
    # 初始化 RAG 引擎
    if 'rag_engine' not in st.session_state:
        with st.spinner("正在加载向量库和模型..."):
            try:
                engine_instance = RAGEngine(txt_path=book_path)
                engine_instance.load_data() 
                st.session_state.rag_engine = engine_instance
                st.success("系统初始化成功！知识库已连接。")
            except Exception as e:
                st.error(f"初始化失败: {str(e)}")
                st.stop()
    
    engine = st.session_state.rag_engine

    st.subheader("知识库管理")
    if st.button("🔄 扫描并更新书籍索引 (断点续传)"):
        with st.spinner("正在处理书籍片段，这可能需要几分钟..."):
            try:
                engine.load_data()
                st.success("索引更新完成！")
            except Exception as e:
                st.error(f"处理出错: {str(e)}")
    
    # 显示状态
    db_path = "./chroma_db"
    is_ready = False

    if os.path.exists(db_path):
        try:
            files = [f for f in os.listdir(db_path) if not f.startswith('.')]
            if len(files) > 0:
                is_ready = True
        except Exception:
            pass

    if is_ready:
        st.success("✅ 知识库已就绪！可以开始提问。")
        if os.path.exists("./checkpoint.json"):
            import json
            with open("./checkpoint.json", 'r', encoding='utf-8') as f:
                cp = json.load(f)
                st.caption(f"上次处理进度: {cp.get('processed_chars', '未知')} 字符")
    else:
        st.warning("⚠️ 尚未检测到知识库。请点击上方按钮扫描书籍。")

    st.markdown("---")
    st.markdown("**关于强势文化**:")
    st.caption("强势文化就是遵循事物规律的文化，弱势文化就是依赖强者的道德期望破格获取的文化。")

# 主聊天区域
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请问关于天道、文化属性或丁元英的任何问题..."):
    # 1. 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答区域
    with st.chat_message("assistant"):
        # 【动态思考状态】
        status_placeholder = st.empty()
        
        thinking_html = """
        <div class="thinking-container">
            <span>🧠 正在深度检索与思考</span>
            <div class="thinking-dots" style="margin-left: 10px;">
                <span></span><span></span><span></span>
            </div>
        </div>
        """
        status_placeholder.markdown(thinking_html, unsafe_allow_html=True)
        
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response_stream = engine.query(prompt)
            has_started_typing = False
            
            if hasattr(response_stream, '__iter__'):
                for chunk in response_stream:
                    if chunk:
                        full_response += chunk
                        
                        if not has_started_typing:
                            status_placeholder.empty()
                            has_started_typing = True
                        
                        message_placeholder.markdown(full_response + "▌")
            else:
                status_placeholder.empty()
                full_response = response_stream
                message_placeholder.markdown(full_response)
            
            # --- 【关键新增】回答结束处理 & 证据链展示 ---
            status_placeholder.empty()
            
            # 渲染最终结果 (移除光标)
            message_placeholder.markdown(full_response)
            
            # === 证据链可视化模块 ===
            # 检查引擎是否有调试数据
            debug_data = getattr(engine, 'last_retrieval_debug_info', [])
            
            if debug_data:
                with st.expander("🔍 点击查看：RAG 证据链与检索诊断", expanded=False):
                    st.caption("💡 **如何阅读**：绿色卡片表示模型在回答中明确引用了该片段；灰色卡片表示检索到了但未被模型使用。这有助于判断是检索不准还是生成忽略。")
                    
                    # 统计指标
                    total_docs = len(debug_data)
                    cited_docs = sum(1 for x in debug_data if x.get("is_cited", False))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("检索片段总数", total_docs)
                    c2.metric("模型引用数", cited_docs)
                    c3.metric("证据利用率", f"{cited_docs/total_docs:.1%}" if total_docs > 0 else "0%")
                    
                    st.markdown("#### 📜 检索片段详情")
                    
                    for item in debug_data:
                        is_cited = item.get("is_cited", False)
                        score = item.get("score", 0)
                        content = item.get("content", "")
                        idx = item.get("index", 0)
                        source = item.get("source", "unknown")
                        
                        # 根据引用状态设置样式类
                        card_class = "cited" if is_cited else "ignored"
                        status_icon = "✅" if is_cited else "⚪"
                        status_text = "已引用" if is_cited else "未引用"
                        
                        # 截断过长的内容用于预览，完整内容可在悬停或点击时看（这里简单展示前300字）
                        preview_content = content[:300] + "..." if len(content) > 300 else content
                        
                        html_card = f"""
                        <div class="evidence-card {card_class}">
                            <div class="evidence-header">
                                <span>{status_icon} 依据 #{idx} | 来源：{os.path.basename(source)}</span>
                                <span class="score-badge">相似度：{score:.4f}</span>
                            </div>
                            <div>{preview_content}</div>
                            <div style="font-size:0.7em; color:#999; margin-top:5px; text-align:right;">状态：{status_text}</div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                    
                    # 智能诊断建议
                    st.markdown("#### 💡 优化建议")
                    if cited_docs == 0:
                        st.error("❌ **生成层失效**：模型检索到了内容，但没有引用任何一条。可能是 Prompt 约束力不足，或者模型产生了幻觉。")
                    elif cited_docs < total_docs * 0.3:
                        st.warning("⚠️ **检索噪音较大**：只有少数片段被使用。建议：1. 减小检索数量 `k`；2. 优化查询重写策略。")
                    elif total_docs < 3:
                        st.info("ℹ️ **检索数量偏少**：总共只找到了很少的片段。建议：1. 增大 `fetch_k`；2. 检查文本切片是否过碎。")
                    else:
                        st.success("✅ **系统运行健康**：检索精准且模型充分利用了上下文证据。")
            else:
                # 如果没有调试数据（可能是旧版 engine），不显示面板或显示提示
                pass
                
        except Exception as e:
            status_placeholder.empty()
            error_msg = f"❌ **生成失败**: {str(e)}\n\n请检查本地 Ollama 服务或模型状态。"
            st.error(error_msg)
            full_response = error_msg 

    # 3. 存入历史 (注意：历史记录中只存纯文本，不存 HTML)
    st.session_state.messages.append({"role": "assistant", "content": full_response})