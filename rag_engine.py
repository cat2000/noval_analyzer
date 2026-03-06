import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ✅ [关键修复] 在此处初始化全局 logger，供 RAGEngine 类使用
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✅ [新增] 自定义轻量级 EnsembleRetriever
# 注意：这里不再依赖外部的 logger 变量，而是自给自足
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Optional

# 为这个类专门创建一个 logger 实例，确保绝对安全
_ensemble_logger = logging.getLogger("SimpleEnsembleRetriever")
# 如果根 logger 没有配置 handler，继承根的配置（通常 basicConfig 已经配好了）
if not _ensemble_logger.handlers:
    _ensemble_logger.setLevel(logging.INFO) 
    # 不需要添加 handler，因为它会自动冒泡到 root logger (由 basicConfig 配置)

class SimpleEnsembleRetriever(BaseRetriever):
    """
    简单的混合检索器，结合多个检索器的结果。
    使用加权排名倒数融合 (Weighted RRF) 策略。
    """
    retrievers: List[BaseRetriever]
    weights: Optional[List[float]] = None
    k: int = 8  # 默认返回数量

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        if not self.retrievers:
            return []
        
        # 1. 获取所有检索器的结果
        all_results = []
        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query) 
                all_results.append(docs)
            except Exception as e:
                # ✅ 关键修复：使用类级别定义的 _ensemble_logger，彻底避开全局 logger 作用域问题
                _ensemble_logger.warning(f"检索器 {retriever} 失败: {e}")
                all_results.append([])
        
        # 2. 计算加权得分 (RRF 策略)
        doc_scores = {}
        
        for i, docs in enumerate(all_results):
            weight = self.weights[i] if self.weights and i < len(self.weights) else 1.0
            
            for rank, doc in enumerate(docs):
                doc_key = doc.page_content
                score = weight / (rank + 1)
                
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"doc": doc, "score": 0.0}
                
                doc_scores[doc_key]["score"] += score
        
        # 3. 按分数排序
        sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # 4. 截取前 k 个
        top_k = self.k if self.k else len(sorted_items)
        return [item["doc"] for item in sorted_items[:top_k]]

class RAGEngine:
    def __init__(self, 
                 txt_path: str, 
                 db_path: str = "./chroma_db", 
                 checkpoint_path: str = "./checkpoint.json",
                 model_name: str = "qwen3", 
                 embed_model_name: str = "bge-large-zh"):
        
        # ✅ 双重保险：在类方法内部也确保 logger 可用
        # 如果文件顶部的 logger 失效，这里会重新获取当前模块的 logger
        global logger
        if 'logger' not in globals() or logger is None:
            logger = logging.getLogger(__name__)

        self.txt_path = txt_path
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        
        # 核心组件初始化
        self.vector_store = None
        self.full_text = ""
        self._data_loaded_attempted = False 
        self.last_retrieval_debug_info = []
        
        # 新增：混合检索器实例
        self.hybrid_retriever = None
        
        # 1. 加载 Embedding 模型
        logger.info(f"正在加载 Embedding 模型: {self.embed_model_name} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )

        # 2. 加载 LLM
        self.llm = OllamaLLM(model=self.model_name, temperature=0.7)
        
        # 3. 文本分割器 (优化版)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len,
            separators=[
                "\n\n第", "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""
            ]
        )
        
        # 预编译章节正则
        self.chapter_pattern = re.compile(r'第\s*(\d+|[一二三四五六七八九十百]+)\s*章')

    def load_data(self):
        """加载数据：读取TXT -> 切分 -> 存入向量库 (支持断点续传)"""
        if self._data_loaded_attempted and self.vector_store is not None:
            return

        self._data_loaded_attempted = True
        
        if not self.full_text:
            self._load_text_content()
            
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
                # 如果有数据但检索器未初始化，尝试初始化检索器
                if not self.hybrid_retriever:
                    self._init_hybrid_retriever()
                return

            logger.info("向量库为空，开始处理文本并生成向量索引...")
            self._process_and_embed()
            
            # 新建库后必须初始化混合检索器
            self._init_hybrid_retriever()
            
        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
            self.vector_store = None
            raise e

    def _init_hybrid_retriever(self):
        """初始化混合检索器 (BM25 + Vector)"""
        if not self.vector_store:
            return
            
        logger.info("正在构建混合检索器 (BM25 + Vector)...")
        try:
            # 1. 获取所有文档用于构建 BM25 索引
            # 注意：如果书非常大，这里可能需要优化内存，但对于单本书通常没问题
            all_docs = self.vector_store.similarity_search("", k=self.vector_store._collection.count())
            
            if not all_docs:
                logger.warning("向量库为空，无法初始化 BM25。")
                return

            # 2. 创建 BM25 检索器
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 8
            
            # 3. 创建向量检索器 (MMR)
            vector_retriever = self.vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5}
            )
            
            # 4. 融合 (权重：BM25 0.5, Vector 0.5)
            # 使用自定义的 SimpleEnsembleRetriever 替代不稳定的库类
            self.hybrid_retriever = SimpleEnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5],
                k=8  # 这里可以控制混合后返回的总数，或者保持默认让上层处理
            )
            logger.info("✅ 混合检索器初始化完成。")
            
        except Exception as e:
            logger.error(f"混合检索器初始化失败: {e}, 将降级为纯向量检索。")
            self.hybrid_retriever = None

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

    def _get_current_chapter(self, text_segment: str, last_chapter: str) -> str:
        """从文本片段中提取最新的章节号"""
        matches = self.chapter_pattern.findall(text_segment)
        if matches:
            # 返回最后一个匹配的章节
            return f"第{matches[-1]}章"
        return last_chapter

    def _process_and_embed(self):
        start_index = self._load_checkpoint()
        if start_index > 0:
            logger.info(f"检测到断点，将从第 {start_index} 个字符处继续处理...")
        
        batch_size = 5000
        total_length = len(self.full_text)
        current_index = start_index
        
        # 追踪当前章节 (用于元数据)
        current_chapter = "序言"
        
        while current_index < total_length:
            end_index = min(current_index + batch_size, total_length)
            chunk_text = self.full_text[current_index:end_index]
            
            # 🔍 更新当前章节状态
            current_chapter = self._get_current_chapter(chunk_text, current_chapter)
            
            docs = self.text_splitter.create_documents([chunk_text])
            
            for doc in docs:
                doc.metadata["source"] = os.path.basename(self.txt_path)
                doc.metadata["start_char"] = current_index
                doc.metadata["chapter"] = current_chapter  # ✅ 写入章节元数据
            
            if docs:
                logger.info(f"正在嵌入批次: {current_index}-{end_index} (章节:{current_chapter}, {len(docs)} 个片段)...")
                self.vector_store.add_documents(docs)
                self._save_checkpoint(end_index)
            
            current_index = end_index
            
            if current_index >= total_length:
                if os.path.exists(self.checkpoint_path):
                    os.remove(self.checkpoint_path)
                logger.info("✅ 所有文本处理完毕，断点文件已清理。")
        
        logger.info("🎉 向量索引构建完成！")

    def query(self, question: str, k: int = 8):
        """执行混合检索增强生成 (RAG)"""
        if not self.vector_store:
            yield "❌ 错误：向量库未初始化。"
            return

        self.last_retrieval_debug_info = []
        search_query = question 
        logger.info(f"[检索] 搜索查询: {search_query}")

        # 🔍 1. 检测是否包含章节约束 (元数据过滤)
        chapter_match = re.search(r'第\s*(\d+|[一二三四五六七八九十百]+)\s*章', question)
        filter_dict = None
        use_hybrid = True

        if chapter_match:
            chapter_name = f"第{chapter_match.group(1)}章"
            logger.info(f"🎯 检测到章节约束: {chapter_name}, 启用元数据过滤。")
            filter_dict = {"chapter": chapter_name}
            # 如果有明确的章节过滤，Chroma 的 filter 与 EnsembleRetriever 配合较复杂
            # 策略：直接使用向量检索 + filter，保证准确性
            use_hybrid = False

        docs = []
        try:
            if use_hybrid and self.hybrid_retriever:
                # 混合检索 (无过滤)
                logger.info("🚀 使用混合检索 (BM25 + Vector)...")
                docs = self.hybrid_retriever.invoke(question)
            else:
                # 纯向量检索 (带过滤 或 无混合检索器)
                logger.info("🎯 使用向量检索 (带元数据过滤)..." if filter_dict else "🎯 使用纯向量检索...")
                docs = self.vector_store.similarity_search(
                    question, 
                    k=k, 
                    filter=filter_dict
                )
                
            # 如果过滤后结果为空，给用户提示
            if not docs and filter_dict:
                yield f"⚠️ 在 {filter_dict['chapter']} 中未找到相关内容，尝试扩大搜索范围..."
                # 降级：去掉过滤再搜一次
                docs = self.vector_store.similarity_search(question, k=k)

        except Exception as e:
            logger.warning(f"检索失败: {e}, 降级为简单相似度搜索。")
            docs = self.vector_store.similarity_search(question, k=k, filter=filter_dict)
        
        if not docs:
            yield "⚠️ 未在知识库中找到相关依据。"
            return

        # 2. 计算相似度得分 (注意：BM25 回来的文档没有向量，需重新计算)
        query_vec = self.embeddings.embed_query(search_query)
        context_evidence = []
        
        for i, d in enumerate(docs):
            # 重新计算向量以获取统一的余弦相似度得分 (用于展示)
            doc_vec = self.embeddings.embed_documents([d.page_content])[0]
            score = float(cosine_similarity([query_vec], [doc_vec])[0][0])
            
            debug_item = {
                "index": i + 1,
                "content": d.page_content,
                "score": round(score, 4),
                "source": d.metadata.get("source", "unknown"),
                "chapter": d.metadata.get("chapter", "未知"), # ✅ 展示章节
                "start_char": d.metadata.get("start_char", 0),
                "is_cited": False
            }
            self.last_retrieval_debug_info.append(debug_item)
            
            # 在上下文中也显示章节信息，帮助 LLM 理解
            evidence_block = f"[依据 {i+1}] (章节:{d.metadata.get('chapter', '?')} | 相似度:{score:.2f}):\n{d.page_content}"
            context_evidence.append(evidence_block)
        
        context_text = "\n\n".join(context_evidence)
        
        # 3. 构建 Prompt
        system_instruction = """
        # Role: 天道规律解析者
        你基于《遥远的救世主》原文进行回答。

        ## 核心指令
        1. **严格依据**: 回答必须完全基于【参考上下文】。
        2. **引用标记**: 使用某段原文时，**必须**标注 `[依据 X]`。
        3. **综合推理**: 多个片段共同说明问题时，标注多个引用，如 `[依据 1][依据 3]`。
        4. **章节感知**: 注意上下文中的章节信息，分析人物思想的变化过程。
        5. **无中生有禁止**: 若无答案，直接说“原文未提及”。

        ## 输出格式
        - 核心观点
        - 详细分析 (融入 `[依据 X]`)
        - 文化属性总结

        ## 参考上下文
        {context}

        ## 用户问题
        {question}

        ## 开始推演：
        """
        
        prompt_text = system_instruction.format(context=context_text, question=question)
        
        # 4. 流式调用 LLM
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

        # 5. 后处理：标记被引用的证据
        cited_indices = set()
        matches = re.findall(r'\[依据\s*(\d+)\]', full_response)
        
        for m in matches:
            try:
                cited_indices.add(int(m))
            except ValueError:
                continue
        
        for item in self.last_retrieval_debug_info:
            if item["index"] in cited_indices:
                item["is_cited"] = True
        
        logger.info(f"[调试] 检索总数:{len(self.last_retrieval_debug_info)}, 引用数:{len(cited_indices)}")