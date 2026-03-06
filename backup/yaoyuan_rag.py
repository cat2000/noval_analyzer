import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM  # 【修复】使用新的独立包
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ================= 配置区域 =================
TEXT_FILE_PATH = "./遥远的救世主.txt"
CHROMA_PERSIST_DIR = "./chroma_db_data"
LLM_MODEL_NAME = "qwen3"  # 请确保你本地 Ollama 已 pull 了这个模型，或者改成你有的模型如 "llama3"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3" # 或者 "m3e-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# ===========================================

def initialize_system():
    print("🚀 正在初始化系统...")
    
    # 1. 加载文档 (修复 UTF-16 编码问题)
    if not os.path.exists(TEXT_FILE_PATH):
        raise FileNotFoundError(f"找不到文件: {TEXT_FILE_PATH}")
    
    print(f"📖 正在加载文件: {TEXT_FILE_PATH} (编码: utf-16)...")
    try:
        loader = TextLoader(TEXT_FILE_PATH, encoding='utf-16')
        documents = loader.load()
        print(f"✅ 成功加载 {len(documents)} 个文档块")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise e

    # 2. 文本切片
    print("🔪 正在分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ 分割完成，共 {len(splits)} 个片段")

    # 3. 初始化 Embedding 模型
    print(f"🧠 正在加载嵌入模型: {EMBEDDING_MODEL_NAME} ...")
    # device="cpu" 强制使用 CPU，如果有 GPU 可改为 "cuda"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 初始化/加载 VectorDB (Chroma)
    print(f"💾 正在连接向量数据库: {CHROMA_PERSIST_DIR} ...")
    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="yaoyuan_collection"
    )

    # 5. 存入数据 (如果数据库是空的)
    if vectordb._collection.count() == 0:
        print("📝 数据库为空，正在存入向量数据 (这可能需要几分钟)...")
        vectordb.add_documents(splits)
        print("✅ 数据存入完成")
    else:
        count = vectordb._collection.count()
        print(f"✅ 数据库已有 {count} 条数据，跳过存入步骤")

    # 6. 初始化 LLM (修复 Ollama 导入)
    print(f"🤖 正在连接大模型: {LLM_MODEL_NAME} ...")
    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.3)

    return vectordb, llm, embeddings

def run_query_with_verification(vectordb, llm, query):
    """执行检索增强生成 (RAG)"""
    
    # 1. 设置检索器
    retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # 每次取最相关的 3 段
    
    # 2. 定义提示词模板 (System Prompt)
    system_prompt = (
        "你是一位精通《遥远的救世主》（电视剧《天道》原著）的智能助手。"
        "请严格根据提供的【上下文】来回答用户的问题。"
        "如果【上下文】中没有包含答案，请直接回答：'抱歉，根据当前资料库，我找不到关于这个问题的具体描述。'"
        "不要编造信息，也不要利用你训练数据中的外部知识来回答书中细节，除非用户明确要求解释概念。"
        "\n\n"
        "【上下文】:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 3. 创建文档组合链 (Stuff 模式)
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # 4. 创建检索链
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    # 5. 执行查询
    result = retrieval_chain.invoke({"input": query})
    return result["answer"]

if __name__ == "__main__":
    try:
        # 初始化系统
        vectordb, llm, embeddings = initialize_system()
        
        print("\n" + "="*50)
        print("🎉 系统启动成功！请输入关于《遥远的救世主》的问题。")
        print("输入 'quit' 退出程序。")
        print("="*50 + "\n")
        
        while True:
            user_input = input("📖 请输入问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                continue
                
            print("🤔 思考中...\n")
            try:
                response = run_query_with_verification(vectordb, llm, user_input)
                print(f"💡 回答: {response}\n")
            except Exception as e:
                print(f"❌ 生成回答时出错: {e}\n")
                
    except Exception as e:
        print(f"💥 系统初始化失败: {e}")
        # 打印详细 traceback 方便调试
        import traceback
        traceback.print_exc()