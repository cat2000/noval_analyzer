import streamlit as st
from rag_engine import RAGEngine
import os
import re
import json

# 页面配置
st.set_page_config(
    page_title="强势文化专家 - 遥远的救世主 RAG (Self-RAG + Rerank)", # ✅ [修改] 标题更新
    page_icon="📚",
    layout="wide"
)

# 自定义 CSS (保持原有样式，增加一点针对 Rerank 分数的样式)
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

    /* 状态提示框 */
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
    
    /* ✅ [新增] Rerank 分数徽章样式 */
    .rerank-badge { 
        background: #e2e6ea; color: #495057; padding: 2px 6px; 
        border-radius: 4px; font-size: 0.7em; margin-left: 5px; 
        border: 1px solid #ced4da; font-weight: bold;
    }
    .rerank-high { background: #d4edda; color: #155724; border-color: #c3e6cb; }
</style>
""", unsafe_allow_html=True)

st.title("📚 强势文化专家系统 (Self-RAG + Rerank)") # ✅ [修改] 标题
st.markdown("基于《遥远的救世主》深度定制。**进阶架构**：LLM 查询重写 + BGE 重排序 + 自我反思生成。")

# 侧边栏
with st.sidebar:
    st.header("控制面板")
    book_path = "./遥远的救世主.txt"
    
    if not os.path.exists(book_path):
        st.error(f"未找到文件：{book_path}")
        st.stop()
    
    # 初始化逻辑
    if 'rag_engine' not in st.session_state:
        with st.spinner("正在加载向量库、Rerank 模型和 LLM (首次运行需下载 BGE-Reranker)..."): # ✅ [修改] 提示语
            try:
                # ✅ [修改] 实例化时不需要额外参数，因为 RAGEngine 有了默认值，但为了明确，可以保持原样
                # 如果 RAGEngine 的 __init__ 强制要求新参数，这里可能需要调整，但目前代码有默认值
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
        
        st.markdown("---")
        # ✅ [修改] 显示更详细的模型配置信息
        eval_model_name = "N/A"
        if hasattr(engine, 'eval_llm') and engine.eval_llm:
            if hasattr(engine.eval_llm, 'model'):
                eval_model_name = engine.eval_llm.model
            elif hasattr(engine.eval_llm, 'model_name'):
                eval_model_name = engine.eval_llm.model_name
            
        rerank_status = "✅ 已加载 (BGE-Large)" if hasattr(engine, 'reranker') and engine.reranker else "❌ 未加载"
        
        st.info(f"**🤖 模型配置**:\n"
                f"- 主模型 (生成/重写): `{engine.model_name}`\n"
                f"- 评估模型 (快速筛选): `{eval_model_name}`\n"
                f"- 重排序模型 (Rerank): `{rerank_status}`")
    else:
        st.warning("⚠️ 尚未检测到知识库。请点击上方按钮。")

    st.markdown("---")
    # ✅ [修改] 更新机制说明
    st.caption("💡 **Self-RAG 机制**：系统会自动评估检索质量。如果结果不佳，将利用 LLM **重写问题**并**重新检索**，直到找到最佳依据。")

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
        
        # ✅ [修改] 更新初始思考状态文案
        thinking_html = """
        <div class="thinking-container">
            <span>🧠 正在执行 Self-RAG 流程 (重写 -> 检索 -> 重排序 -> 反思)</span>
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
                    # 1. 处理错误/警告信息
                    if chunk.startswith("⚠️") or chunk.startswith("❌"):
                        status_placeholder.markdown(f"<div class='crag-status-info'>{chunk}</div>", unsafe_allow_html=True)
                        continue
                    
                    # 2. 处理心跳信号
                    if chunk.startswith("💡"):
                        status_placeholder.markdown(
                            "<div class='crag-status-info' style='background-color:#d4edda; border-color:#28a745; color:#155724;'>✅ 检索与重排序完成，正在深度推演...</div>", 
                            unsafe_allow_html=True
                        )
                        continue 
                    
                    # 3. 正常内容处理
                    if not has_started_typing:
                        status_placeholder.empty()
                        has_started_typing = True
                    
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            # 渲染最终结果
            status_placeholder.empty()
            if full_response:
                message_placeholder.markdown(full_response)
            else:
                if not status_placeholder._is_empty: 
                     pass
                else:
                    message_placeholder.markdown("⚠️ 未能生成有效回答，请检查日志。")
            
            # === 证据链可视化 ===
            debug_data = getattr(engine, 'last_retrieval_debug_info', [])
            
            if debug_data:
                with st.expander("🔍 点击查看：RAG 证据链与 Self-RAG 诊断", expanded=False): # ✅ [修改] 标题
                    st.caption("💡 **绿色卡片**：模型引用了该片段。 **灰色卡片**：检索到但未被引用。所有卡片均经过 **BGE Rerank** 重排序。")
                    
                    total_docs = len(debug_data)
                    cited_docs = sum(1 for x in debug_data if x.get("is_cited", False))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("重排序后片段", total_docs)
                    c2.metric("模型实际引用", cited_docs)
                    c3.metric("证据利用率", f"{cited_docs/total_docs:.1%}" if total_docs > 0 else "0%")
                    
                    st.markdown("#### 📜 检索片段详情 (按 Rerank 分数排序)")
                    
                    for item in debug_data:
                        is_cited = item.get("is_cited", False)
                        score = item.get("score", 0) # 向量相似度
                        content = item.get("content", "")
                        idx = item.get("index", 0)
                        chapter = item.get("chapter", "未知章节")
                        
                        # ✅ [修改] 获取 Rerank 分数 (新字段)
                        rerank_score = item.get("rerank_score", 0) 
                        
                        card_class = "cited" if is_cited else "ignored"
                        status_icon = "✅" if is_cited else "⚪"
                        
                        # ✅ [修改] 根据 Rerank 分数动态决定标签
                        if rerank_score > 0.8:
                            rerank_label = f"🔥 极高相关 ({rerank_score:.2f})"
                            rerank_class = "rerank-high"
                        elif rerank_score > 0.5:
                            rerank_label = f"🟢 高相关 ({rerank_score:.2f})"
                            rerank_class = ""
                        else:
                            rerank_label = f"❄️ 参考 ({rerank_score:.2f})"
                            rerank_class = ""
                        
                        preview_content = content[:300] + "..." if len(content) > 300 else content
                        
                        html_card = f"""
                        <div class="evidence-card {card_class}">
                            <div class="evidence-header">
                                <div>
                                    <span>{status_icon} 依据 #{idx}</span>
                                    <span class="chapter-badge">📖 {chapter}</span>
                                    <!-- ✅ [修改] 显示 Rerank 分数 -->
                                    <span class="rerank-badge {rerank_class}">{rerank_label}</span>
                                </div>
                                <span class="score-badge">向量相似度：{score:.4f}</span>
                            </div>
                            <div>{preview_content}</div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                    
                    # 智能诊断
                    st.markdown("#### 💡 系统诊断")
                    
                    if total_docs == 0:
                        st.error("❌ **检索彻底失败**：即使经过 Self-RAG 重试，仍未找到相关片段。")
                    elif cited_docs == 0:
                        st.warning("⚠️ **生成层失效**：检索到了相关片段（Rerank 高分），但模型未引用任何内容。可能是 Prompt 约束过严。")
                    elif cited_docs < total_docs * 0.3:
                        st.info(f"ℹ️ **分析提示**：模型引用了 {cited_docs} 个核心片段。还有 {total_docs - cited_docs} 个 [参考] 片段未被直接使用。这通常是正常的，因为 Self-RAG 已经过滤掉了低质内容，剩余的是作为背景补充。")
                    else:
                        st.success("✅ **深度分析模式**：模型充分利用了多个高分片段进行综合解读。")

        except Exception as e:
            status_placeholder.empty()
            error_msg = f"❌ **系统错误**: {str(e)}"
            st.error(error_msg)
            full_response = error_msg 

    st.session_state.messages.append({"role": "assistant", "content": full_response})