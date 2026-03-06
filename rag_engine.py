import os
import json
import logging
import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, 
                 txt_path: str, 
                 db_path: str = "./chroma_db", 
                 checkpoint_path: str = "./checkpoint.json",
                 model_name: str = "qwen3", 
                 embed_model_name: str = "bge-large-zh"):
        
        self.txt_path = txt_path
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        
        # 核心组件初始化
        self.vector_store = None
        self.full_text = ""
        self._data_loaded_attempted = False 
        self.last_retrieval_debug_info = []  # ✅ 证据链暂存区
        
        # 1. 加载 Embedding 模型 (BGE-Large-ZH)
        logger.info(f"正在加载 Embedding 模型: {self.embed_model_name} ...")
        # 可选：设置镜像加速 (如果网络慢)
        # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",
            model_kwargs={'device': 'cpu'}, # 有 GPU 改为 'cuda'
            encode_kwargs={
                'normalize_embeddings': True, # ✅ 必须归一化以计算余弦相似度
                'batch_size': 32
            }
        )

        # 2. 加载 LLM (Ollama)
        self.llm = OllamaLLM(model=self.model_name, temperature=0.7)
        
        # 3. 文本分割器 (优化版)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,       # 增大片段以包含更多情节
            chunk_overlap=300,     # 增大重叠保持连贯
            length_function=len,
            separators=[
                "\n\n第",          # 优先在章节标题前切分
                "\n\n",            
                "\n",              
                "。", "！", "？",   
                "；", "，", " ", ""                
            ]
        )
        
    def load_data(self):
        """加载数据：读取TXT -> 切分 -> 存入向量库 (支持断点续传)"""
        if self._data_loaded_attempted and self.vector_store is not None:
            return

        self._data_loaded_attempted = True
        
        # 1. 读取文本
        if not self.full_text:
            self._load_text_content()
            
        # 2. 连接向量库
        logger.info("正在初始化向量数据库连接...")
        try:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            
            current_docs_count = self.vector_store._collection.count()
            logger.info(f"当前向量库中已有 {current_docs_count} 个文档片段。")
            
            if current_docs_count > 0:
                logger.info("检测到已有向量数据，连接成功。")
                return

            # 3. 空库则构建索引
            logger.info("向量库为空，开始处理文本并生成向量索引...")
            self._process_and_embed()
            
        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
            self.vector_store = None
            raise e
        
    def _load_text_content(self):
        """读取 TXT 文件，处理编码"""
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"找不到书籍文件: {self.txt_path}")
        
        logger.info(f"正在读取书籍: {os.path.basename(self.txt_path)} ...")
        encodings_to_try = ['utf-16', 'utf-8', 'gbk']
        
        for encoding in encodings_to_try:
            try:
                with open(self.txt_path, 'r', encoding=encoding) as f:
                    self.full_text = f.read()
                logger.info(f"✅ 成功通过 {encoding} 读取文件。")
                return
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"编码 {encoding} 尝试失败: {e}")
                continue
                
        raise ValueError(f"无法读取文件 {self.txt_path}。")

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("processed_chars", 0)
            except Exception as e:
                logger.warning(f"⚠️ 断点文件损坏 ({e})，将重置。")
                try: os.remove(self.checkpoint_path)
                except: pass
                return 0
        return 0

    def _save_checkpoint(self, processed_chars: int):
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"processed_chars": processed_chars}, f)
        except Exception as e:
            logger.error(f"保存断点失败: {e}")

    def _process_and_embed(self):
        start_index = self._load_checkpoint()
        if start_index > 0:
            logger.info(f"检测到断点，将从第 {start_index} 个字符处继续处理...")
        
        batch_size = 5000
        total_length = len(self.full_text)
        current_index = start_index
        
        while current_index < total_length:
            end_index = min(current_index + batch_size, total_length)
            chunk_text = self.full_text[current_index:end_index]
            
            docs = self.text_splitter.create_documents([chunk_text])
            for i, doc in enumerate(docs):
                doc.metadata["source"] = os.path.basename(self.txt_path)
                # 注意：这里的 start_char 是相对于当前 batch 的，实际全局位置需累加，但用于展示足够
                doc.metadata["start_char"] = current_index 
            
            if docs:
                logger.info(f"正在嵌入批次: {current_index}-{end_index} ({len(docs)} 个片段)...")
                self.vector_store.add_documents(docs)
                self._save_checkpoint(end_index)
            
            current_index = end_index
            
            if current_index >= total_length:
                if os.path.exists(self.checkpoint_path):
                    os.remove(self.checkpoint_path)
                logger.info("✅ 所有文本处理完毕，断点文件已清理。")
        
        logger.info("🎉 向量索引构建完成！")

    def query(self, question: str, k: int = 8):
        """
        执行检索增强生成 (RAG)，并记录证据链调试信息。
        """
        if not self.vector_store:
            yield "❌ 错误：向量库未初始化。请先在侧边栏点击'扫描并更新书籍索引'。"
            return

        # 1. 初始化调试容器
        self.last_retrieval_debug_info = []
        
        # 2. 查询重写 (简单版：直接使用原问题，也可扩展为 LLM 重写)
        # 为了演示证据链，这里直接使用用户问题作为搜索查询
        search_query = question 
        logger.info(f"[检索] 搜索查询: {search_query}")

        # 3. 执行检索 (MMR 策略)
        docs = []
        try:
            docs = self.vector_store.max_marginal_relevance_search(
                search_query, 
                k=k,               
                fetch_k=40,        # 候选池扩大
                lambda_mult=0.5    
            )
        except Exception as e:
            logger.warning(f"MMR 搜索失败，降级为相似度搜索: {e}")
            docs = self.vector_store.similarity_search(search_query, k=k)
        
        if not docs:
            yield "⚠️ 未在知识库中找到相关依据，无法进行深度推演。"
            return

        # 4. 计算相似度得分并构建上下文
        # 预计算 Query 向量
        query_vec = self.embeddings.embed_query(search_query)
        
        context_evidence = []
        
        for i, d in enumerate(docs):
            # 计算 Document 向量
            doc_vec = self.embeddings.embed_documents([d.page_content])[0]
            # 计算余弦相似度
            score = float(cosine_similarity([query_vec], [doc_vec])[0][0])
            
            # 记录调试信息
            debug_item = {
                "index": i + 1,
                "content": d.page_content,
                "score": round(score, 4),
                "source": d.metadata.get("source", "unknown"),
                "start_char": d.metadata.get("start_char", 0),
                "is_cited": False # 默认未引用
            }
            self.last_retrieval_debug_info.append(debug_item)
            
            # 构建给 LLM 的上下文块 (强制要求标记引用来源)
            evidence_block = f"[依据 {i+1}] (相似度:{score:.2f}):\n{d.page_content}"
            context_evidence.append(evidence_block)
        
        context_text = "\n\n".join(context_evidence)
        
        # 5. 构建 Prompt (包含强约束指令)
        system_instruction = """
        # Role: 天道规律解析者 (Expert of The Way)
        你基于《遥远的救世主》原文进行回答。

        ## 核心指令
        1. **严格依据**: 你的回答必须完全基于提供的【参考上下文】。
        2. **引用标记**: 当你使用某段原文作为依据时，**必须**在句末或段落中标注 `[依据 X]`，其中 X 是对应片段的编号。
        3. **综合推理**: 不要只依赖单一片段。如果多个片段共同说明了一个问题，请综合它们，并标注多个引用，如 `[依据 1][依据 3]`。
        4. **无中生有禁止**: 如果上下文中没有答案，请直接说“原文未提及”，不要编造。

        ## 输出格式
        - 先列出核心观点。
        - 展开分析，并在分析中自然融入 `[依据 X]`。
        - 最后总结文化属性规律。

        ## 参考上下文
        {context}

        ## 用户问题
        {question}

        ## 开始推演：
        """
        
        prompt_text = system_instruction.format(context=context_text, question=question)
        
        # 6. 流式调用 LLM
        logger.info("[生成] 正在基于事实进行逻辑推演...")
        full_response = ""
        
        try:
            for chunk in self.llm.stream(prompt_text):
                if chunk:
                    full_response += chunk
                    yield chunk
        except Exception as e:
            logger.error(f"LLM 推演失败: {e}")
            yield f"\n\n❌ **逻辑推演中断**: {str(e)}"
            return

        # 7. 【后处理】分析 LLM 回复，标记被引用的证据
        # 正则匹配 [依据 1], [依据 12] 等
        cited_indices = set()
        matches = re.findall(r'\[依据\s*(\d+)\]', full_response)
        
        for m in matches:
            try:
                idx = int(m)
                cited_indices.add(idx)
            except ValueError:
                continue
        
        # 更新调试列表状态
        for item in self.last_retrieval_debug_info:
            if item["index"] in cited_indices:
                item["is_cited"] = True
        
        logger.info(f"[调试] 检索总数:{len(self.last_retrieval_debug_info)}, 引用数:{len(cited_indices)}")
        # 此时 self.last_retrieval_debug_info 已更新，前端可读取