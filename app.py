import streamlit as st
from rag_engine import RAGEngine
import os

# 页面配置
st.set_page_config(
    page_title="强势文化专家 - 遥远的救世主 RAG",
    page_icon="📚",
    layout="wide"
)

# 自定义 CSS 美化 (包含动态呼吸灯动画)
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
    
    /* --- 新增：动态呼吸点动画 --- */
    @keyframes blink {
        0%, 100% { opacity: 0.2; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    .thinking-dots span {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin: 0 3px;
        background-color: #0068c9; /* Streamlit 蓝色 */
        border-radius: 50%;
        animation: blink 1.4s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(1) { animation-delay: 0s; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    /* 思考中的容器样式 */
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
    

    # 初始化 RAG 引擎 (单例模式模拟)
    if 'rag_engine' not in st.session_state:
        with st.spinner("正在加载向量库和模型..."):
            try:
                # 1. 创建引擎实例
                engine_instance = RAGEngine(txt_path=book_path)
                
                # 2. 【关键修复】无条件调用 load_data
                # 让 engine 内部判断是连接现有库还是新建
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
                            status_placeholder.empty() # 移除思考动画
                            has_started_typing = True
                        
                        # 打字过程中保留光标
                        message_placeholder.markdown(full_response + "▌")
            else:
                status_placeholder.empty()
                full_response = response_stream
                message_placeholder.markdown(full_response)
            
            # --- 【关键新增】回答结束处理 ---
            status_placeholder.empty() # 确保思考栏彻底消失
            
            # 构建最终显示内容：正文 + 结束提示
            # 使用较小的字体和绿色图标，不喧宾夺主
            end_marker = "\n\n---\n<span style='color: #28a745; font-size: 0.85em;'>✅ 回答完毕，如有疑问请继续提问。</span>"
            
            # 渲染最终结果 (移除光标，加上结束标记)
            message_placeholder.markdown(full_response + end_marker, unsafe_allow_html=True)
            
        except Exception as e:
            status_placeholder.empty()
            error_msg = f"❌ **生成失败**: {str(e)}\n\n请检查本地 Ollama 服务或模型状态。"
            st.error(error_msg)
            full_response = error_msg 

    # 3. 存入历史 (注意：历史记录中只存纯文本，不存 HTML 结束标记，以免下次加载乱码)
    st.session_state.messages.append({"role": "assistant", "content": full_response})