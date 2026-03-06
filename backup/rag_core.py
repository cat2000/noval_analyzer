import os
from typing import Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import config
from prompt_engine import get_prompt_version_v3_ultimate

class TianDaoRAGSystem:
    """
    天道 RAG 系统核心类
    负责：文档加载、智能拆分、向量库构建、查询推理
    """
    def __init__(self):
        self.vectordb: Optional[Chroma] = None
        self.llm: Optional[OllamaLLM] = None
        self._initialized = False

    def initialize(self):
        """初始化系统组件"""
        if self._initialized:
            return

        # 1. 检查文件
        if not os.path.exists(config.DATA_FILE):
            raise FileNotFoundError(f"❌ 未找到小说文件: {config.DATA_FILE}\n请将文件放入 data/ 目录")

        # 2. 加载文档 (兼容多种编码)
        documents = []
        encodings = ['utf-16', 'utf-8', 'gbk']
        for enc in encodings:
            try:
                loader = TextLoader(config.DATA_FILE, encoding=enc)
                documents = loader.load()
                print(f"✅ 成功加载文件 (编码: {enc})")
                break
            except Exception:
                continue
        
        if not documents:
            raise ValueError("❌ 无法读取文件，请检查文件编码或路径")

        # 3. 【关键】合理拆分文档
        # 使用递归字符拆分器，按段落、句子层级切割，保持语义完整
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""]
        )
        splits = splitter.split_documents(documents)
        print(f"🔪 文档已智能拆分：共 {len(splits)} 个片段 (每段约{config.CHUNK_SIZE}字)")

        # 4. 初始化 Embedding 和 VectorDB
        print(f"🧠 加载嵌入模型: {config.EMBEDDING_MODEL} ...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectordb = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="yaoyuan_collection"
        )

        # 5. 存入数据 (如果库为空)
        if self.vectordb._collection.count() == 0:
            print("📝 数据库为空，正在构建向量索引 (首次运行可能需要几分钟)...")
            self.vectordb.add_documents(splits)
            print("✅ 索引构建完成")
        else:
            count = self.vectordb._collection.count()
            print(f"✅ 数据库已有 {count} 条数据，跳过构建步骤。")

        # 6. 初始化 LLM
        print(f"🤖 连接大模型: {config.LLM_MODEL} ...")
        self.llm = OllamaLLM(model=config.LLM_MODEL, temperature=0.3)
        
        self._initialized = True
        print("🚀 系统初始化完毕")

    def query(self, question: str) -> str:
        """执行查询"""
        if not self._initialized:
            self.initialize()
        
        if not self.vectordb or not self.llm:
            raise RuntimeError("系统未正确初始化")

        # 创建检索器
        retriever = self.vectordb.as_retriever(search_kwargs={"k": config.SEARCH_K})
        
        # 获取 Prompt
        prompt_template = get_prompt_version_v3_ultimate()
        
        # 创建组合链
        combine_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt_template)
        
        # 创建最终检索链
        retrieval_chain = create_retrieval_chain(retriever, combine_chain)
        
        # 执行并返回结果
        result = retrieval_chain.invoke({"input": question})
        return result["answer"]

# 全局单例实例
rag_system = TianDaoRAGSystem()