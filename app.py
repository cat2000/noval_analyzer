import streamlit as st
from rag_engine import RAGEngine
import os
import re
import json

# 页面配置
st.set_page_config(
    page_title="强势文化专家 - 遥远的救世主 RAG (CRAG 加速版)",
    page_icon="📚",
    layout="wide"
)

# 自定义 CSS
st.markdown("""
<style>
    .stChatMessage { font-family: 'Georgia', serif; }
    .expert-response { border-left: 4px solid #d9534f; padding-left: 10px; background-color: #f9f9f9; }
    
    /* 动态呼吸点动画 */
    @keyframes blink {
        0%, 100% { opacity: 0.2; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    .thinking-dots span {
        display: inline-block; width: 8px; height: 8px; margin: 0 3px;
        background-color: #0068c9; border-radius: 50%;
        animation: blink 1.4s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(1) { animation-delay: 0s; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    .thinking-container {
        display: flex; align-items: center; color: #0068c9; font-weight: 600;
        padding: 12px 16px; background-color: #e8f4fd; border-radius: 8px;
        border-left: 5px solid #0068c9; margin-bottom: 10px;
        font-family: sans-serif; font-size: 0.95rem;
    }

    /* CRAG 状态提示框 */
    .crag-status-info {
        background-color: #fff3cd; border-left: 5px solid #ffc107;
        padding: 10px; margin-bottom: 10px; border-radius: 4px;
        color: #856404; font-family: sans-serif; font-size: 0.9rem;
        animation: fadeIn 0.3s;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }

    /* 证据链卡片 */
    .evidence-card {
        border-left: 5px solid #ccc; background-color: #f8f9fa;
        padding: 12px; margin-bottom: 12px; border-radius: 4px;
        font-family: monospace; font-size: 0.85rem; white-space: pre-wrap;
        line-height: 1.4; transition: all 0.2s;
    }
    .evidence-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .evidence-card.cited { border-left-color: #2ecc71; background-color: #f0fff4; }
    .evidence-card.ignored { border-left-color: #bdc3c7; background-color: #f9f9f9; opacity: 0.85; }
    
    .evidence-header {
        display: flex; justify-content: space-between; align-items: center;
        font-size: 0.75rem; color: #555; margin-bottom: 8px;
        font-family: sans-serif; font-weight: bold; flex-wrap: wrap; gap: 5px;
    }
    .chapter-badge { background: #0068c9; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; font-weight: normal; }
    .score-badge { background: #eee; color: #333; padding: 2px 6px; border-radius: 4px; }
    .crag-badge { 
        background: #d1ecf1; color: #0c5460; padding: 2px 6px; 
        border-radius: 4px; font-size: 0.7em; margin-left: 5px; 
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 强势文化专家系统 (CRAG 加速版)")
st.markdown("基于《遥远的救世主》深度定制。**双模型协同**：小模型极速评估 + 大模型深度推理。")

# 侧边栏
with st.sidebar:
    st.header("控制面板")
    book_path = "./遥远的救世主.txt"
    
    if not os.path.exists(book_path):
        st.error(f"未找到文件：{book_path}")
        st.stop()
    
    # 初始化逻辑
    if 'rag_engine' not in st.session_state:
        with st.spinner("正在加载向量库和模型 (首次运行需下载 BGE)..."):
            try:
                engine_instance = RAGEngine(txt_path=book_path)
                engine_instance.load_data() 
                st.session_state.rag_engine = engine_instance
                st.success("系统初始化成功！")
            except Exception as e:
                st.error(f"初始化失败: {str(e)}")
                st.stop()
    
    engine = st.session_state.rag_engine

    st.subheader("知识库管理")
    if st.button("🔄 重新扫描书籍索引"):
        with st.spinner("正在处理书籍片段..."):
            try:
                engine._data_loaded_attempted = False 
                engine.load_data()
                st.success("索引更新完成！")
                st.rerun()
            except Exception as e:
                st.error(f"处理出错: {str(e)}")
    
    # 状态显示
    db_path = "./chroma_db"
    is_ready = os.path.exists(db_path) and any(os.scandir(db_path))

    if is_ready:
        st.success("✅ 知识库已就绪！")
        if os.path.exists("./checkpoint.json") and engine.full_text:
            try:
                with open("./checkpoint.json", 'r', encoding='utf-8') as f:
                    cp = json.load(f)
                    processed = cp.get('processed_chars', 0)
                    total = len(engine.full_text)
                    if total > 0:
                        percent = f"{(processed/total*100):.1f}%"
                        st.caption(f"处理进度: {processed}/{total} ({percent})")
            except Exception:
                pass
        
        # ✅ [修改点 1] 安全地获取模型名称
        st.markdown("---")
        eval_model_name = "N/A"
        if hasattr(engine, 'eval_llm') and engine.eval_llm:
            # 尝试获取 model 属性，兼容不同版本的 langchain_ollama
            if hasattr(engine.eval_llm, 'model'):
                eval_model_name = engine.eval_llm.model
            elif hasattr(engine.eval_llm, 'model_name'):
                eval_model_name = engine.eval_llm.model_name
            
        st.info(f"**🤖 模型配置**:\n- 主模型 (生成): `{engine.model_name}`\n- 评估模型 (CRAG): `{eval_model_name}`")
    else:
        st.warning("⚠️ 尚未检测到知识库。请点击上方按钮。")

    st.markdown("---")
    st.caption("💡 **CRAG 加速机制**：使用多线程 + 小模型并行评估检索片段，将等待时间从秒级降低到毫秒级。")

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
        
        # 初始思考状态
        thinking_html = """
        <div class="thinking-container">
            <span>🧠 正在并行检索与评估 (CRAG 加速模式)</span>
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
            
            # ✅ [修改点 2] 简化迭代逻辑，并处理“心跳信号”
            for chunk in response_stream:
                if chunk:
                    # 1. 处理错误/警告信息 (显示在状态栏，不进入正文)
                    if chunk.startswith("⚠️") or chunk.startswith("❌"):
                        status_placeholder.markdown(f"<div class='crag-status-info'>{chunk}</div>", unsafe_allow_html=True)
                        continue
                    
                    # 2. 处理心跳信号 (💡 开头)
                    # 你的 rag_engine 会先 yield "💡 正在结合原文推演...\n\n"
                    # 我们这里选择：如果是心跳，只更新状态栏提示“开始生成”，不把它写入最终回复
                    if chunk.startswith("💡"):
                        status_placeholder.markdown(
                            "<div class='crag-status-info' style='background-color:#d4edda; border-color:#28a745; color:#155724;'>✅ 检索完成，正在生成回答...</div>", 
                            unsafe_allow_html=True
                        )
                        # 注意：这里没有 continue，如果你想让这句话也显示在对话框里，就去掉下面的 if not has_started_typing 逻辑
                        # 但通常我们不希望正文里有一句“正在推演”，所以这里我们选择不将其加入 full_response，或者直接跳过
                        # 方案 A: 跳过不显示在正文 (推荐)
                        continue 
                    
                    # 3. 正常内容处理
                    if not has_started_typing:
                        status_placeholder.empty() # 清空状态栏
                        has_started_typing = True
                    
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            # 渲染最终结果
            status_placeholder.empty()
            if full_response:
                message_placeholder.markdown(full_response)
            else:
                # 如果 full_response 为空，且没有错误提示，说明可能流被完全过滤了
                if not status_placeholder._is_empty: 
                     pass # 状态栏已有错误信息
                else:
                    message_placeholder.markdown("⚠️ 未能生成有效回答，请检查日志。")
            
            # === 证据链可视化 ===
            debug_data = getattr(engine, 'last_retrieval_debug_info', [])
            
            if debug_data:
                with st.expander("🔍 点击查看：RAG 证据链与 CRAG 诊断", expanded=False):
                    st.caption("💡 **绿色卡片**：模型引用了该片段。 **灰色卡片**：检索到但未被引用。所有卡片均已通过 CRAG 相关性评估。")
                    
                    total_docs = len(debug_data)
                    cited_docs = sum(1 for x in debug_data if x.get("is_cited", False))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("通过评估的片段", total_docs)
                    c2.metric("模型实际引用", cited_docs)
                    c3.metric("证据利用率", f"{cited_docs/total_docs:.1%}" if total_docs > 0 else "0%")
                    
                    st.markdown("#### 📜 检索片段详情")
                    
                    for item in debug_data:
                        is_cited = item.get("is_cited", False)
                        score = item.get("score", 0)
                        content = item.get("content", "")
                        idx = item.get("index", 0)
                        chapter = item.get("chapter", "未知章节")
                        
                        # ✅ [新增] 获取 CRAG 真实评分
                        crag_score = item.get("crag_score", 0) 
                        
                        card_class = "cited" if is_cited else "ignored"
                        status_icon = "✅" if is_cited else "⚪"
                        
                        # ✅ [新增] 根据评分动态决定标签文案和颜色
                        if crag_score == 1:
                            crag_label = "🔥 高相关"
                            crag_color_style = "background: #d4edda; color: #155724; border-color: #c3e6cb;" # 绿色
                        else:
                            crag_label = "❄️ 参考/保守保留"
                            crag_color_style = "background: #f8f9fa; color: #6c757d; border-color: #dae0e5;" # 灰色
                        
                        preview_content = content[:300] + "..." if len(content) > 300 else content
                        
                        html_card = f"""
                        <div class="evidence-card {card_class}">
                            <div class="evidence-header">
                                <div>
                                    <span>{status_icon} 依据 #{idx}</span>
                                    <span class="chapter-badge">📖 {chapter}</span>
                                    <!-- ✅ [修改] 动态插入样式和文字 -->
                                    <span class="crag-badge" style="{crag_color_style}">{crag_label}</span>
                                </div>
                                <span class="score-badge">相似度：{score:.4f}</span>
                            </div>
                            <div>{preview_content}</div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                    
                    # 智能诊断
                    st.markdown("#### 💡 系统诊断")
                    
                    if total_docs == 0:
                        st.error("❌ **检索完全失败**：没有文档通过 CRAG 评估。")
                    elif cited_docs == 0:
                        st.warning("⚠️ **生成层失效**：检索到了相关片段，但模型未引用任何内容。")
                    elif cited_docs < total_docs * 0.3:
                        # ✅ [修改] 调整文案，不再暗示这是“噪音”，而是提示“可挖掘空间”
                        st.info(f"ℹ️ **分析提示**：模型引用了 {cited_docs} 个核心片段。还有 {total_docs - cited_docs} 个 [参考] 片段未被直接使用，可能是因为模型认为当前回答已足够深入，或者这些片段仅作为背景存在。如需更深入分析，请尝试追问“请结合更多细节分析”。")
                    else:
                        st.success("✅ **深度分析模式**：模型充分利用了多个片段进行综合解读。")

        except Exception as e:
            status_placeholder.empty()
            error_msg = f"❌ **系统错误**: {str(e)}"
            st.error(error_msg)
            full_response = error_msg 

    st.session_state.messages.append({"role": "assistant", "content": full_response})