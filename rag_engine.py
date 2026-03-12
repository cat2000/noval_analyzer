import os
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain & AI libs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# ✅ 改用本地文件存储，实现持久化
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Logger 配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalFileStore:
    """
    一个简单的本地文件存储实现，用于存储 Parent-Child 映射关系。
    替代 langchain.storage.LocalFileStore 或 langchain_core.storage.BaseStore
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def _get_path(self, key: str) -> str:
        # 防止 key 中包含非法文件名字符
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(self.path, f"{safe_key}.json")

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """批量获取"""
        results = []
        for key in keys:
            path = self._get_path(key)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        # 假设存的是 Document 的 dict 形式，这里直接读出来
                        # 如果存的是序列化后的 Document 对象，可能需要根据实际保存格式调整
                        data = json.load(f)
                        results.append(data)
                except Exception:
                    results.append(None)
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: List[tuple]) -> None:
        """批量设置"""
        for key, value in key_value_pairs:
            path = self._get_path(key)
            try:
                # 如果 value 是 Document 对象，通常需要序列化
                # 但看你的代码逻辑，存入的是 p_doc (Document 对象)
                # 我们需要确保它能被 json 序列化。
                # 如果 Document 对象不能直接 json.dump，需要转换。
                # 这里做一个简单的兼容处理：如果是 Document，转为 dict
                
                data_to_save = value
                if hasattr(value, 'page_content'): # 判断是否是 Document 对象
                    # 简单序列化 Document
                    data_to_save = {
                        "page_content": value.page_content,
                        "metadata": value.metadata,
                        "type": "Document" # 标记类型，方便以后还原（如果需要）
                    }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False)
            except Exception as e:
                logger.error(f"保存文档到本地存储失败 {key}: {e}")

    def mdelete(self, keys: List[str]) -> None:
        """批量删除"""
        for key in keys:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """遍历所有 key"""
        if not os.path.exists(self.path):
            return
        for filename in os.listdir(self.path):
            if filename.endswith(".json"):
                key = filename[:-5] # 去掉 .json
                if prefix is None or key.startswith(prefix):
                    yield key

# ==========================================
# 工具类：Pipeline 性能计时器
# ==========================================
class PipelineTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.stages = {}

    def checkpoint(self, stage_name: str) -> float:
        current_time = time.time()
        duration = current_time - self.last_checkpoint
        self.stages[stage_name] = round(duration, 4)
        self.last_checkpoint = current_time
        logger.info(f"⏱️ [{stage_name}] 耗时：{duration:.4f} 秒")
        return duration

    def get_total_time(self) -> float:
        return time.time() - self.start_time

    def get_stages(self) -> Dict[str, float]:
        return self.stages

# ==========================================
# 流式生成收集器 (保持原有逻辑，确保流式稳定)
# ==========================================
class StreamingResponseCollector:
    def __init__(self, llm, prompt_text):
        self.llm = llm
        self.prompt_text = prompt_text
        self.full_response = ""
        self.first_line_processed = False
        self.first_line_buffer = ""
        self.MAX_FIRST_LINE_LEN = 80
        self.question_for_dedup = ""
        
    def generate(self, question=""):
        """生成流并累积内容"""
        self.question_for_dedup = question
        try:
            stream_generator = self.llm.stream(self.prompt_text)
            
            for chunk in stream_generator:
                if not chunk:
                    continue
                
                self.full_response += chunk
                
                # 首行去重逻辑
                if not self.first_line_processed:
                    self.first_line_buffer += chunk
                    should_process = '\n' in self.first_line_buffer or len(self.first_line_buffer) >= self.MAX_FIRST_LINE_LEN
                    
                    if should_process:
                        self.first_line_processed = True
                        parts = self.first_line_buffer.split('\n', 1) if '\n' in self.first_line_buffer else [self.first_line_buffer, ""]
                        line = parts[0].strip()
                        remainder = parts[1] if len(parts) > 1 else ""
                        
                        is_duplicate = False
                        if self.question_for_dedup:
                            q_clean = re.sub(r'[?？,.，!！:\:]', '', self.question_for_dedup).strip()
                            t_clean = re.sub(r'^#+\s*', '', line).strip()
                            t_clean = re.sub(r'[?？,.，!！:\:]', '', t_clean).strip()
                            if len(q_clean) > 2 and len(t_clean) > 2:
                                if q_clean == t_clean: is_duplicate = True
                                elif (q_clean in t_clean or t_clean in q_clean) and abs(len(q_clean) - len(t_clean)) < 5: is_duplicate = True
                        
                        if not is_duplicate:
                            output = line + "\n" + remainder if remainder else line + "\n"
                            yield output
                        elif remainder:
                            yield remainder
                else:
                    yield chunk
            
            if not self.first_line_processed and self.first_line_buffer:
                line = self.first_line_buffer.strip()
                is_duplicate = False
                if self.question_for_dedup:
                    q_clean = re.sub(r'[?？,.，!！]', '', self.question_for_dedup).strip()
                    t_clean = re.sub(r'^#+\s*', '', line).strip()
                    t_clean = re.sub(r'[?？,.，!！]', '', t_clean).strip()
                    if len(q_clean) > 2 and len(t_clean) > 2 and q_clean == t_clean:
                        is_duplicate = True
                if not is_duplicate:
                    yield line + "\n\n"
        except Exception as e:
            logger.error(f"流式生成过程中出错：{e}")
            raise e

# ==========================================
# 组件：自定义混合检索器
# ==========================================
class SimpleEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: Optional[List[float]] = None
    k: int = 8

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        if not self.retrievers:
            return []
        all_results = []
        for retriever in self.retrievers:
            try:
                if hasattr(retriever, 'k'):
                    retriever.k = self.k
                docs = retriever.invoke(query)
                all_results.append(docs)
            except Exception as e:
                logger.warning(f"检索器 {retriever} 失败：{e}")
                all_results.append([])
        
        doc_scores = {}
        for i, docs in enumerate(all_results):
            weight = self.weights[i] if self.weights and i < len(self.weights) else 1.0
            for rank, doc in enumerate(docs):
                doc_key = doc.page_content
                score = weight / (rank + 1)
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"doc": doc, "score": 0.0}
                doc_scores[doc_key]["score"] += score
        
        sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        top_k = self.k if self.k else len(sorted_items)
        return [item["doc"] for item in sorted_items[:top_k]]

# ==========================================
# 核心引擎：RAGEngine (✅ 修复版：回归分数优先策略)
# ==========================================
class RAGEngine:
    PROMPT_REWRITE_EXAMPLES = """
    示例 1:
    用户："什么是天道？"
    重写：《遥远的救世主》中丁元英对"天道"的定义，以及"天道"与"文化属性"、"强势文化"和"弱势文化"之间的逻辑关系。
    示例 2:
    用户："丁元英为什么那么厉害？"
    重写：丁元英在《遥远的救世主》中展现出的超凡认知能力来源，他对"文化属性"规律的掌握，以及他如何利用这些规律在商战和人性博弈中取胜的具体案例分析。
    """

    def __init__(self, 
                 txt_path: str, 
                 db_path: str = "./chroma_db", 
                 checkpoint_path: str = "./checkpoint.json",
                 model_name: str = "qwen3", 
                 eval_model_name: str = "qwen3:0.6b",
                 embed_model_name: str = "bge-large-zh",
                 rerank_model_name: str = "BAAI/bge-reranker-large"):
        
        self.txt_path = txt_path
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        today_str = datetime.now().strftime('%Y%m%d')
        self.log_file = os.path.join(self.log_dir, f"rag_session_{today_str}.jsonl")
        logger.info(f"📊 可观测性日志已启用，保存路径：{self.log_file}")

        self.vector_store = None
        self.full_text = ""
        self._data_loaded_attempted = False 
        self.last_retrieval_debug_info = []
        self.hybrid_retriever = None
        self.metrics = {}
        
        # 默认参数 (允许前端覆盖)
        self.default_params = {
            "top_k_initial": 6,      
            "top_k_final": 2,
            "max_self_rag_attempts": 1, # 
            "multi_query_count": 1,     # 
            "rerank_threshold": 0.5,
            "eval_top_k": 3  # ✅ 新增：评估时考察前 K 个文档            
        }

        self._load_models(embed_model_name, rerank_model_name, model_name, eval_model_name)

        # ✅ 修改点 1: 定义父子切分器
        # 父文档：保持完整语境 (约 2000 字)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", "。", "！", "？"]
        )
        # 子文档：用于精准检索 (约 400 字)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=50, 
            separators=["\n\n", "\n", "。", "！", "？"]
        )

        # ✅ 修改点 2: 初始化文档存储 (LocalFileStore 持久化到 ./doc_store 目录)
        # 这样重启后无需重新生成摘要和父子映射
        self.doc_store = LocalFileStore("./doc_store")
        
        # 标记是否已构建父子索引
        self.parent_child_index_built = False

        # 原有的 text_splitter 不再直接用于最终检索，但可保留用于兼容或调试
        self.text_splitter = self.child_splitter 
        self.chapter_pattern = re.compile(r'第\s*(\d+|[一二三四五六七八九十百]+)\s*章')
        

    def _load_models(self, embed_name, rerank_name, llm_name, eval_llm_name):
        logger.info(f"正在加载 Embedding 模型：{embed_name} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        logger.info(f"正在加载 Reranker 模型：{rerank_name} ...")
        try:
            self.reranker = CrossEncoder(model_name=rerank_name, max_length=512, device='cpu')
            logger.info("✅ Reranker 加载成功。")
        except Exception as e:
            logger.error(f"❌ Reranker 加载失败：{e}，将降级为不使用 Rerank。")
            self.reranker = None

        logger.info(f"正在加载主模型：{llm_name} ...")
        self.llm = OllamaLLM(model=llm_name, temperature=0.35, request_timeout=120, num_predict=2048)
        
        self.eval_llm = self.llm
        if eval_llm_name and eval_llm_name != llm_name:
            try:
                logger.info(f"正在加载评估小模型：{eval_llm_name} ...")
                self.eval_llm = OllamaLLM(model=eval_llm_name, temperature=0, request_timeout=60)
            except Exception as e:
                logger.warning(f"加载评估模型失败，降级使用主模型。")

    def load_data(self):
        if self._data_loaded_attempted and self.vector_store is not None:
            return
        self._data_loaded_attempted = True
        if not self.full_text:
            self._load_text_content()
            
        logger.info("正在初始化向量数据库...")
        try:
            self.vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            current_docs_count = self.vector_store._collection.count()
            logger.info(f"向量库现有片段：{current_docs_count}")
            
            if current_docs_count > 0:
                if not self.hybrid_retriever:
                    self._init_hybrid_retriever()
                return

            logger.info("向量库为空，开始构建索引...")
            self._process_and_embed()
            self._init_hybrid_retriever()
        except Exception as e:
            logger.error(f"向量库初始化失败：{e}")
            raise e

    def _init_hybrid_retriever(self):
        if not self.vector_store: return
        
        logger.info("构建自定义父子文档检索器 (Manual Parent-Child Mapping)...")
        try:
            # 1. 获取所有子文档用于构建 BM25
            all_data = self.vector_store.get(include=["metadatas", "documents"])
            if not all_data['documents']:
                logger.warning("向量库为空，无法构建检索器。")
                return

            sub_docs = [Document(page_content=c, metadata=m) for c, m in zip(all_data['documents'], all_data['metadatas'])]
            
            # 2. 创建 BM25 检索器 (基于子文档)
            bm25_retriever = BM25Retriever.from_documents(sub_docs)
            bm25_retriever.k = self.default_params["top_k_initial"] * 2
            
            # 3. ✅ 创建自定义的向量检索器 (替代 ParentDocumentRetriever)
            # 逻辑：先检索子文档，然后根据 parent_id 从 doc_store 加载父文档
            vector_store_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.default_params["top_k_initial"]}
            )
            
            # 4. ✅ 包装成一个能自动返回父文档的检索器
            class ParentChildWrapper(BaseRetriever):
                vector_retriever: BaseRetriever
                doc_store: Any
                k: int = 8
                
                def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                    # A. 先检索子文档
                    sub_docs = self.vector_retriever.invoke(query)
                    
                    # B. 提取唯一的 parent_id
                    parent_ids = set()
                    for doc in sub_docs:
                        pid = doc.metadata.get("parent_id")
                        if pid:
                            parent_ids.add(pid)
                    
                    # C. 从 doc_store 批量加载父文档
                    parent_docs = []
                    seen_parents = set()
                    for pid in parent_ids:
                        if pid not in seen_parents:
                            raw_data = self.doc_store.mget([pid])[0]
                            if raw_data:
                                # ✅ 重要：如果读出来是 dict，需还原为 Document 对象
                                if isinstance(raw_data, dict) and "page_content" in raw_data:
                                    from langchain_core.documents import Document
                                    p_doc = Document(
                                        page_content=raw_data.get("page_content", ""),
                                        metadata=raw_data.get("metadata", {})
                                    )
                                else:
                                    p_doc = raw_data # 已经是 Document 对象或其他
                                parent_docs.append(p_doc)
                                seen_parents.add(pid)
                    
                    # D. 如果没有找到父文档（极端情况），降级返回子文档
                    if not parent_docs:
                        logger.warning("未找到父文档，降级返回子文档。")
                        return sub_docs[:self.k]
                    
                    # E. 返回父文档 (限制数量)
                    return parent_docs[:self.k]

            # 实例化包装器
            vector_parent_retriever = ParentChildWrapper(
                vector_retriever=vector_store_retriever,
                doc_store=self.doc_store,
                k=self.default_params["top_k_initial"]
            )
            
            # 5. ✅ 最终策略：只使用这个自定义的父子检索器
            # (如果你非常想混合 BM25，可以把 vector_parent_retriever 和 bm25_retriever 传入 SimpleEnsembleRetriever)
            # 但为了稳定性，这里先只用向量检索的父子模式
            self.hybrid_retriever = vector_parent_retriever
            
            logger.info("✅ 自定义父子文档检索器就绪 (手动映射父文档)。")
            
        except Exception as e:
            logger.error(f"检索器初始化失败：{e}")
            # 降级方案：直接返回向量库检索子文档
            self.hybrid_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.default_params["top_k_initial"]})

    def _load_text_content(self):
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"找不到书籍文件：{self.txt_path}")
        encodings_to_try = ['utf-16', 'utf-8', 'gbk']
        for encoding in encodings_to_try:
            try:
                with open(self.txt_path, 'r', encoding=encoding) as f:
                    self.full_text = f.read()
                logger.info(f"✅ 读取成功 ({encoding})。")
                return
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法读取文件。")

    def _process_and_embed(self):
        """
        ✅ 支持断点续传的父子文档索引构建
        """
        logger.info("🚀 开始构建父子文档索引与摘要元数据...")
        
        # 1. 先按父文档粒度切分 (内存操作，很快)
        parent_docs = self.parent_splitter.create_documents([self.full_text])
        total_parents = len(parent_docs)
        logger.info(f"共规划 {total_parents} 个父文档片段。")

        # 2. 检查断点
        start_index = self._load_checkpoint()
        if start_index >= 0:
            logger.info(f"📍 检测到断点：上次成功处理到索引 {start_index}。")
            logger.info(f"⏩ 将从索引 {start_index + 1} 继续处理...")
            # 验证 chroma_db 中是否真的有数据，防止文件不同步
            current_count = self.vector_store._collection.count()
            if current_count == 0:
                logger.warning("⚠️ 警告：Checkpoint 存在但向量库为空！可能数据不一致，建议删除 checkpoint.json 后重试。")
                # 这里选择保守策略：如果库为空，强制从头开始，避免数据错乱
                start_index = -1
            else:
                # 跳过已处理的文档列表，节省内存
                # 注意：parent_docs 是完整的，我们只是跳过循环
                pass
        else:
            logger.info("🆕 未检测到有效断点，将从头开始构建。")
            start_index = -1

        docs_to_embed = []
        total_children = 0
        
        # 如果是续传，我们需要知道之前已经生成了多少个子文档（可选，仅用于日志美观）
        if start_index >= 0:
            # 简单估算或直接设为 0，因为我们是追加写入 chroma
            logger.info(f"之前已生成的子文档将保留在向量库中，本次仅追加新数据。")

        # 3. 遍历父文档
        for i, p_doc in enumerate(parent_docs):
            # ✅ 断点续传核心：跳过已处理的索引
            if i <= start_index:
                continue

            # --- 方案三：生成摘要元数据 ---
            try:
                logger.info(f"📝 [进度 {i+1}/{total_parents}] 正在为第 {i+1} 个父文档生成摘要...") 
                
                summary_prompt = (
                    f"请阅读以下《遥远的救世主》片段，用一句话（30 字以内）概括其核心思想或情节：\n"
                    f"{p_doc.page_content[:500]}"
                )
                
                start_gen = time.time()
                summary = self.eval_llm.invoke(summary_prompt).strip()
                logger.info(f"✅ 第 {i+1} 个文档摘要生成完成，耗时：{time.time() - start_gen:.2f}秒")
                
                if "核心思想：" in summary: summary = summary.split("核心思想：")[1]
                if "概括：" in summary: summary = summary.split("概括：")[1]
            except Exception as e:
                logger.error(f"❌ 生成摘要失败 (Index {i}): {e}")
                # 关键决策：如果 LLM 失败，是重试还是跳过？
                # 这里选择跳过并标记，避免死循环卡住整个进程
                summary = "生成失败：无法连接模型或超时"
                # 也可以选择 raise e 来停止程序，让用户手动修复后重试
            
            # 更新父文档元数据
            p_doc.metadata.update({
                "source": os.path.basename(self.txt_path),
                "summary": summary,
                "parent_id": f"parent_{i}",
                "is_parent": True
            })
            
            current_chapter = self._get_current_chapter(p_doc.page_content, "未知章节")
            p_doc.metadata["chapter"] = current_chapter

            # 将父文档存入 Doc Store (Key-Value 存储)
            # 这一步很快，本地文件 IO
            try:
                self.doc_store.mset([(f"parent_{i}", p_doc)])
            except Exception as e:
                logger.error(f"保存父文档到本地存储失败：{e}")

            # --- 方案一：切分子文档 ---
            child_docs = self.child_splitter.split_documents([p_doc])
            
            for c_doc in child_docs:
                c_doc.metadata.update({
                    "parent_id": f"parent_{i}",
                    "summary": summary,
                    "chapter": current_chapter,
                    "source": os.path.basename(self.txt_path),
                    "is_parent": False
                })
                docs_to_embed.append(c_doc)
            
            total_children += len(child_docs)
            
            # 4. 批量嵌入并保存 (每处理完一个父文档就写入，确保断点准确)
            if docs_to_embed:
                try:
                    # 单个父文档的子文档数量不多，直接 add 即可
                    self.vector_store.add_documents(docs_to_embed)
                    docs_to_embed = [] # 清空缓存
                except Exception as e:
                    logger.error(f"嵌入向量失败：{e}")
                    # 如果嵌入失败，不要保存 checkpoint，下次重启会重试这个文档
                    continue

            # 5. ✅ 保存断点 (关键：只有当 doc_store 和 vector_store 都成功后才更新)
            self._save_checkpoint(i)
            
            # 进度日志
            if (i + 1) % 10 == 0 or i == total_parents - 1:
                logger.info(f"🎉 阶段性完成：已处理 {i+1}/{total_parents} 父文档，累计子文档 {self.vector_store._collection.count()} (含历史)。")

        # 6. 清理 Checkpoint (全部完成后)
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            logger.info("🧹 所有文档处理完毕，已清除 Checkpoint 文件。")
        
        self.parent_child_index_built = True
        logger.info("🎉 父子文档索引与摘要元数据构建完成！")

    def _load_checkpoint(self) -> int:
        """
        读取断点信息。
        返回: 最后一个已成功处理的 parent_index (下次应从 index + 1 开始)
        """
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 兼容旧格式
                    if "last_processed_index" in data:
                        return data["last_processed_index"]
                    # 如果没有新字段，尝试估算（旧逻辑可能不准，建议直接返回 -1 重头来或手动清理）
                    # 这里为了安全，如果格式不对，返回 -1 表示从头开始
                    logger.warning("Checkpoint 格式过旧，将重新构建索引。")
                    return -1
            except Exception as e:
                logger.error(f"读取 Checkpoint 失败：{e}，将重新构建。")
                return -1
        return -1

    def _save_checkpoint(self, index: int):
        """
        保存断点信息。
        参数: index - 当前刚刚成功处理完的 parent_index
        """
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"last_processed_index": index}, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存 Checkpoint 失败：{e}")

    def _get_current_chapter(self, text_segment: str, last_chapter: str) -> str:
        """
        增强版章节提取：
        1. 使用更宽松的正则匹配多种格式（中文数字、阿拉伯数字、全角数字、Chapter X）。
        2. 优先扫描文本前 600 字（标题通常在片段开头）。
        3. 【关键】如果当前片段没找到，且 last_chapter 有效，则直接继承上一章。
           这解决了 RecursiveCharacterTextSplitter 把标题切到上一个片段的问题。
        """
        if not text_segment:
            return last_chapter

        # 定义更强大的正则模式
        # 匹配：第 1 章，第 一 章，第 1 节，Chapter 1, 一章
        pattern = re.compile(
            r'(?:第\s*([0-90-9一二三四五六七八九十百千]+)\s*[章节])|'  # 第 X 章/节
            r'(?:Chapter\s*([0-9]+))|'                              # Chapter X
            r'(?:([一二三四五六七八九十百千]+)\s*章)'                # X 章 (如 "一章")
        )
        
        # 只扫描前 600 个字符，提高效率且避免匹配到正文中出现的"第 X 章"字样
        search_text = text_segment[:600] 
        matches = pattern.findall(search_text)
        
        if matches:
            # findall 返回元组列表，例如 [('1', '', ''), ('', '2', '')]
            # 我们需要取第一个非空的分组
            first_match = matches[0]
            chapter_num = ""
            for group in first_match:
                if group:
                    chapter_num = group
                    break
            
            if chapter_num:
                return f"第{chapter_num}章"
        
        # 【核心修复】如果当前片段没找到章节号：
        # 检查是否有上一个有效的章节信息。如果有，说明这是同一章的后续部分，直接继承。
        if last_chapter and last_chapter != "未知章节":
            return last_chapter
            
        return "未知章节"

    def _log_interaction(self, question: str, full_response: str, metrics: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response_length": len(full_response),
            "metrics": metrics,
            "retrieval_snapshot": [
                {
                    "idx": d["index"], "chapter": d["chapter"], "rerank_score": d["rerank_score"],
                    "is_cited": d["is_cited"], "content_preview": d["content"][:200] + "..." if len(d["content"]) > 200 else d["content"]
                } for d in self.last_retrieval_debug_info
            ],
            "response_preview": full_response[:500] + "..." if len(full_response) > 500 else full_response,
            "risk_flags": {
                "hallucination_risk": metrics.get("cited_count", 0) == 0 and metrics.get("total_retrieved", 0) > 0,
                "low_utilization": metrics.get("citation_rate", 0) < 0.3 and metrics.get("total_retrieved", 0) > 2,
                "high_latency": metrics.get("latency_seconds", 0) > 15.0
            }
        }
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"❌ 日志记录失败：{e}")

    # ==========================================
    # 核心逻辑方法
    # ==========================================
    
    def _rewrite_query(self, question: str, history_context: str = "", mode: str = "deep") -> str:
        if mode == "direct":
            return question
        if mode == "light":
            clean_q = re.sub(r'(请问 | 帮我 | 分析一下 | 什么是 | 为什么)', '', question)
            return clean_q.strip()
        
        if len(question) > 40 and any(k in question for k in ["具体情节", "原文片段", "第几章"]):
            return question

        prompt = (
            f"你是一个精通《遥远的救世主》的资深研究助手。\n"
            f"任务：将用户问题重写为**语义丰富、包含关键实体**的检索查询。\n\n"
            f"## 参考范例:\n{self.PROMPT_REWRITE_EXAMPLES}\n\n"
            f"## 当前任务:\n"
            f"用户问题：{question}\n"
            f"{f'之前尝试过的相关背景：{history_context}' if history_context else ''}\n\n"
            f"请直接输出重写后的查询，不要额外解释。\n重写："
        )
        try:
            response = self.eval_llm.invoke(prompt).strip()
            if "重写:" in response:
                response = response.split("重写:", 1)[1].strip()
            elif "重写：" in response:
                response = response.split("重写：", 1)[1].strip()
            return response.strip('"\'')
        except Exception as e:
            logger.warning(f"语义重写失败，回退到原问题：{e}")
            return question

    def _generate_multi_queries_parallel(self, question: str, n: int = 1) -> List[str]:
        base_prompt = f"你是一个检索专家。请基于用户关于《遥远的救世主》的问题，生成一个不同角度的检索查询。\n用户问题：{question}\n请直接输出查询语句，不要额外解释："
        queries = []
        max_workers = min(n, 5) 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.llm.invoke, base_prompt) for _ in range(n)]
            for future in as_completed(futures):
                try:
                    resp = future.result().strip()
                    if resp: queries.append(resp)
                except Exception as e:
                    logger.warning(f"并行生成查询失败：{e}")
        
        queries = list(dict.fromkeys(queries))
        if question not in queries: queries.append(question)
        return queries[:n+1]

    def _rerank_docs(self, query: str, docs: List[Document], top_k_final: int) -> List[Document]:
        if not self.reranker or not docs:
            return docs[:top_k_final]
        
        logger.info(f"🔍 正在对 {len(docs)} 个片段进行 Rerank...")
        pairs = [[query, doc.page_content] for doc in docs]
        
        try:
            scores = self.reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
            if hasattr(scores, 'tolist'): scores = scores.tolist()
            elif not isinstance(scores, list): scores = [float(scores)] * len(docs)
            
            for i, doc in enumerate(docs):
                doc.metadata['rerank_score'] = scores[i] if isinstance(scores, list) else float(scores)
            
            ranked_docs = sorted(docs, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)
            return ranked_docs[:top_k_final] 
        except Exception as e:
            logger.error(f"Rerank 出错：{e}, 返回原始顺序。")
            return docs[:top_k_final]

    def _build_context_and_metrics(self, final_docs, question, timer):
        """✅ 优化点：批量计算 Embedding，消除 N+1 问题"""
        context_evidence = []
        timer.checkpoint("Context_Embedding_Start") 
        
        try:
            # 1. 只计算一次 Question 向量
            q_vec = self.embeddings.embed_query(question)
            
            # 2. 批量计算所有 Document 向量
            doc_contents = [d.page_content for d in final_docs]
            d_vecs = self.embeddings.embed_documents(doc_contents)
            
            # 3. 批量计算余弦相似度
            similarities = cosine_similarity([q_vec], d_vecs)[0]
        except Exception as e:
            logger.warning(f"批量嵌入计算失败：{e}")
            similarities = [0.0] * len(final_docs)
        
        for i, d in enumerate(final_docs):
            score = similarities[i] if i < len(similarities) else 0.0
            r_score = d.metadata.get('rerank_score', 0)
            tag_label = "🔥[核心依据]" if r_score > 0.7 else "❄️[辅助参考]"
        
            # ✅ 新增：提取摘要
            summary = d.metadata.get('summary', '无摘要')
            chapter = d.metadata.get('chapter', '?')
            
            tag_label = "🔥[核心依据]" if d.metadata.get('rerank_score', 0) > 0.7 else "❄️[辅助参考]"
            
            # 记录调试信息
            self.last_retrieval_debug_info.append({
                "index": i + 1, 
                "content": d.page_content, 
                "summary": summary, # 记录摘要
                "score": round(score, 4),
                "rerank_score": round(d.metadata.get('rerank_score', 0), 4), 
                "chapter": chapter, 
                "is_cited": False
            })
            
            # ✅ 修改 Context 格式：加入摘要作为引导
            context_segment = (
                f"[依据 {i+1}] {tag_label} (章节:{chapter})\n"
                f"> 📝 **核心摘要**: {summary}\n"
                f"{d.page_content}"
            )
            context_evidence.append(context_segment)
        
        timer.checkpoint("Context_Building_Done") 
        return "\n\n---\n\n".join(context_evidence)

    def _generate_response_stream(self, context_text, question):
        # ✅ 新版 Prompt：兼顾“丁元英式干练”与“文雅排版”
        prompt_template = """
# Role: 《遥远的救世主》资深研究专家
你兼具丁元英的**洞察力**与学者的**文雅**。请基于【参考上下文】回答用户问题。

## ⚡ 核心原则
1. **直击本质**：开篇第一句直接给出核心结论。**严禁**铺垫、寒暄、重复问题或使用“根据上下文”、“综上所述”等废话。
2. **文雅排版**：
   - 关键概念、金句必须 **加粗**。
   - 引用原文必须使用 > 引用块 格式，并在末尾标注章节，如：`—— [第X章]`。
   - 多点论述使用简洁的无序列表 (-)，每项尽量简短有力。
3. **严谨举证 (强制执行)**：
   - **每一句**基于原文的观点、数据或情节描述，**必须**在句末明确标注来源索引，格式为 `(见依据 N)`，其中 N 是参考上下文中对应的编号 (1, 2, 3...)。
   - **禁止**只写章节号而不写依据编号。章节号用于展示出处，`(见依据 N)` 用于定位具体片段，两者**缺一不可**。
   - 示例：丁元英的核心认知源于对文化属性的洞察 `(见依据 1)`。他在古城的经历验证了这一理论 `(见依据 2)`。
4. **极简主义**：能用一句话说清的，绝不用两段。拒绝车轱辘话。

## 参考上下文
{context}

## 用户问题
{question}

## 你的回答 (立即开始，注意排版美观，切记标注依据编号):
"""
        prompt_text = prompt_template.format(context=context_text, question=question)
        
        # 初始化收集器
        collector = StreamingResponseCollector(self.llm, prompt_text)
        
        # 返回收集器和生成器
        return collector, collector.generate(question)

    # ==========================================
    # ✅ 核心修复：回归“分数优先”的快速评估逻辑
    # ==========================================
    def _evaluate_retrieval_quality(self, question: str, docs: List[Document]) -> bool:
        """
        ✅ 优化版评估逻辑：
        1. Top-K 宽容度：检查前 eval_top_k 个文档中是否有至少一个相关。
        2. 启发式分数过滤：
           - 计算前 K 个文档的平均 Rerank 分数。
           - 高分 (>0.75) -> 直接通过 (True)。
           - 低分 (<0.45) -> 直接失败 (False)。
           - 中间分 -> 调用 LLM 对前 K 个进行批量评估。
        """
        if not docs:
            return False
        
        # 获取配置参数
        eval_k = self.default_params.get("eval_top_k", 3)
        # 确保不超过实际文档数
        k_to_eval = min(len(docs), eval_k)
        docs_to_check = docs[:k_to_eval]
        
        # 提取 Rerank 分数
        scores = [d.metadata.get('rerank_score', 0) for d in docs_to_check]
        
        if not scores:
            return False
            
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        logger.info(f"📊 [评估] 前 {k_to_eval} 个文档 - 平均分：{avg_score:.2f}, 最高分：{max_score:.2f}")

        # --- 策略 1: 基于聚合分数的快速通道 (Heuristic Fast Path) ---
        
        # 情况 A: 平均分很高，说明整体质量极好，直接通过
        if avg_score > 0.75:
            logger.info(f"✅ [快模式] 平均分 ({avg_score:.2f}) > 0.75，直接判定检索成功。")
            return True
        
        # 情况 B: 最高分都很低，说明最好的也不咋地，直接失败
        # 注意：这里用 max_score 比 avg_score 更严格，防止漏掉唯一的相关项，但也防止全是噪音
        if max_score < 0.45:
            logger.warning(f"⚠️ [快模式] 最高分 ({max_score:.2f}) < 0.45，直接判定检索失败。")
            return False
        
        # --- 策略 2: 中间地带，调用 LLM 进行 Top-K 批量评估 ---
        logger.info(f"🤔 [慢模式] 分数处于中间地带 ({avg_score:.2f})，启动 LLM 辅助评估前 {k_to_eval} 个文档...")
        
        # 构造 Prompt：一次性让 LLM 判断这 K 个文档里有没有相关的
        # 这样比循环调用 K 次 LLM 要快得多，且能利用 LLM 的比较能力
        docs_context = ""
        for i, doc in enumerate(docs_to_check):
            docs_context += f"[文档 {i+1}]: {doc.page_content[:200]}...\n"
        
        prompt = (
            f"你是一个严格的检索评估专家。\n"
            f"用户问题：{question}\n\n"
            f"系统检索到了以下 {k_to_eval} 个候选片段：\n{docs_context}\n\n"
            f"任务：判断上述片段中，**是否存在至少一个**片段包含回答用户问题所需的**实质性信息**？\n"
            f"注意：不需要所有片段都相关，只要有一个有用即可。\n\n"
            f"如果存在相关片段，输出 'YES'。\n"
            f"如果所有片段都无关或全是噪音，输出 'NO'。\n"
            f"只输出 YES 或 NO，不要解释。"
        )
        
        try:
            resp = self.eval_llm.invoke(prompt).strip().upper()
            
            if 'YES' in resp:
                logger.info("✅ LLM 判定：前 K 个文档中存在相关内容，评估通过。")
                return True
            else:
                logger.warning("⚠️ LLM 判定：前 K 个文档均不相关，评估失败，触发重试。")
                return False
                
        except Exception as e:
            logger.error(f"❌ LLM 评估调用失败：{e}，保守起见返回 True (不阻断流程)。")
            return True

    def _post_process_response(self, full_response, context_text, question, timer):
        """后处理：统计引用 (增强版：支持章节号模糊匹配)"""
        
        def count_citations(text: str):
            indices = set()
            
            # 1. 优先匹配严格的依据编号：(见依据 1), [依据 2], (1) 等
            pattern_strict = r'[\[\(【](?:依据 | 参考 | 见依据 | 见参考)?\s*:?\s*(\d+)[\]\)】]'
            matches_strict = re.findall(pattern_strict, text)
            
            for m in matches_strict:
                try: 
                    idx = int(m)
                    # 确保索引在有效范围内
                    if 1 <= idx <= len(self.last_retrieval_debug_info):
                        indices.add(idx)
                except ValueError: 
                    pass
            
            # 2. 【兜底策略】如果严格匹配数量为 0，尝试匹配章节号进行模糊关联
            # 场景：LLM 只写了 "—— [第四章]"，没写 "(见依据 1)"
            if len(indices) == 0 and len(self.last_retrieval_debug_info) > 0:
                logger.info("⚠️ 未检测到严格依据编号，尝试通过章节号进行模糊匹配...")
                
                # 匹配 "—— [第 X 章]" 或 "—— [第四章]"
                chapter_pattern = r'——\s*\[第\s*([0-90-9一二三四五六七八九十百]+)\s*章\]'
                chapter_matches = re.findall(chapter_pattern, text)
                
                if chapter_matches:
                    # 遍历所有检索到的依据，看章节是否匹配
                    for item in self.last_retrieval_debug_info:
                        item_chapter = item.get('chapter', '') # 例如："第 4 章"
                        
                        for q_chapter_num in chapter_matches:
                            # 简单归一化：如果检索结果的章节包含用户输出的章节数字/文字
                            # 注意：这里是一个简化逻辑，完美方案需要中文数字转阿拉伯数字
                            if str(q_chapter_num) in item_chapter or item_chapter in f"第{q_chapter_num}章":
                                indices.add(item["index"])
                                logger.debug(f"模糊匹配成功：依据 {item['index']} ({item_chapter}) <-> 文中提到 [第{q_chapter_num}章]")
            
            return indices, len(indices)

        cited_indices, cited_count = count_citations(full_response)
        
        # 如果连模糊匹配都没找到，才判定为缺少引用
        needs_retry = (cited_count == 0 and len(self.last_retrieval_debug_info) > 0)
        
        if needs_retry:
            logger.warning("⚠️ [自检警告] 生成结果完全缺少引用标识 (含章节号)。添加系统注记。")
            full_response += "\n\n> ⚠️ **系统注**: 本次回答未能自动标注原文引用或章节出处，请人工核对上下文。"
        
        # 更新 debug_info 中的 is_cited 状态，供前端显示
        for item in self.last_retrieval_debug_info:
            if item["index"] in cited_indices:
                item["is_cited"] = True
            else:
                item["is_cited"] = False # 显式设为 False
        
        return full_response, cited_count, cited_indices, needs_retry

    def query(self, question: str, **kwargs):
        """
        主查询入口 (✅ 修复版)
        """
        params = {**self.default_params, **kwargs}
        rewrite_mode = params.get('rewrite_mode', 'deep')
        top_k_initial = params['top_k_initial']
        top_k_final = params['top_k_final']
        multi_query_count = params['multi_query_count']
        max_attempts = params.get('max_self_rag_attempts', 0) # 默认 0

        logger.info(f"🎛️ [本次配置] K_init={top_k_initial}, Mode={rewrite_mode}, Max_Retry={max_attempts}")

        timer = PipelineTimer()
        timer.checkpoint("Start")

        if not self.vector_store:
            yield "❌ 错误：向量库未初始化。"
            return

        self.last_retrieval_debug_info = []
        final_docs = []
        retry_overhead_seconds = 0.0 
        attempt = 0
        success_eval = False

        # === Self-RAG 循环 (带智能降级) ===
        while attempt <= max_attempts and not success_eval:
            is_retry = (attempt > 0)
            logger.info(f"🔄 [尝试 {attempt + 1}/{max_attempts + 1}] {'(降级重试)' if is_retry else '(初始检索)'}")
            
            try:
                # 1. 准备查询
                if is_retry:
                    current_queries = [question]
                    logger.info("⚡ [降级策略] 重试模式：禁用多路查询，直接使用原问题。")
                    current_top_k = min(int(top_k_initial * 1.5), 20) 
                else:
                    rewritten_query = self._rewrite_query(question, mode=rewrite_mode) 
                    timer.checkpoint("Query_Rewrite")
                    
                    if rewrite_mode != 'direct' and multi_query_count > 0:
                        multi_vars = self._generate_multi_queries_parallel(question, n=multi_query_count)
                        timer.checkpoint("MultiQuery_Gen")
                        current_queries = list(dict.fromkeys([question, rewritten_query] + multi_vars))
                    else:
                        current_queries = [rewritten_query if rewritten_query else question]
                    current_top_k = top_k_initial

                # 2. 并行检索
                all_docs = []
                seen_content = set()
                
                def retrieve_single_query(q):
                    try:
                        if self.hybrid_retriever:
                            old_k = self.hybrid_retriever.k
                            self.hybrid_retriever.k = current_top_k
                            res = self.hybrid_retriever.invoke(q)
                            self.hybrid_retriever.k = old_k
                            return res
                        else:
                            return self.vector_store.similarity_search(q, k=current_top_k)
                    except Exception as e:
                        logger.warning(f"查询 '{q}' 检索失败：{e}")
                        return []

                with ThreadPoolExecutor(max_workers=len(current_queries)) as executor:
                    futures = {executor.submit(retrieve_single_query, q): q for q in current_queries}
                    for future in as_completed(futures):
                        docs = future.result()
                        for doc in docs:
                            if doc.page_content not in seen_content:
                                seen_content.add(doc.page_content)
                                all_docs.append(doc)
                
                timer.checkpoint("Parallel_Retrieval")

                # 3. Rerank
                if all_docs:
                    ranked_docs = self._rerank_docs(question, all_docs, top_k_final)
                    timer.checkpoint("Rerank")
                    final_docs = ranked_docs
                else:
                    final_docs = []
                
                # 4. 评估 (✅ 使用修复后的快速评估)
                if final_docs:
                    success_eval = self._evaluate_retrieval_quality(question, final_docs)
                    timer.checkpoint("Eval_Phase")
                    
                    if not success_eval and attempt < max_attempts:
                        logger.warning("❌ 评估未通过，准备重试...")
                        attempt += 1
                        retry_overhead_seconds += timer.get_total_time() - sum(timer.get_stages().values())
                        continue
                    elif not success_eval:
                        logger.warning("❌ 评估未通过且已达最大重试次数，强制继续生成。")
                else:
                    if attempt < max_attempts:
                        attempt += 1
                        continue
                    else:
                        logger.warning("❌ 无检索结果且已达最大重试次数。")

            except Exception as e:
                logger.error(f"检索阶段发生异常：{e}")
                if attempt < max_attempts:
                    attempt += 1
                    continue
                final_docs = []
                break
            
            break 

        timer.checkpoint("Retrieval_Phase_Done")
        
        if not final_docs:
            yield "⚠️ 经过多次检索与评估，未找到足够的原文依据来回答这个问题。"
            self.metrics = {
                "status": "failed", "latency_seconds": round(timer.get_total_time(), 2),
                "stage_durations": timer.get_stages(), "total_retrieved": 0, "cited_count": 0,
                "config_snapshot": params, "attempts_made": attempt + 1
            }
            self._log_interaction(question, "", self.metrics)
            return

        # 5. 构建上下文
        context_text = self._build_context_and_metrics(final_docs, question, timer)
        
        # 6. LLM 生成
        logger.info("📝 开始生成并自我反思...")
        
        full_response = ""
        try:
            collector, stream_gen = self._generate_response_stream(context_text, question)
            
            for chunk in stream_gen:
                if chunk:
                    yield chunk
            
            full_response = collector.full_response
            
        except Exception as e:
            logger.error(f"生成流异常：{e}")
            yield f"\n\n❌ 生成中断：{str(e)}"
            return

        timer.checkpoint("LLM_Generation")
        
        # 7. 后处理
        full_response, cited_count, cited_indices, needs_retry = self._post_process_response(full_response, context_text, question, timer)
        
        # 8. 组装 Metrics
        total_latency = timer.get_total_time()
        current_stages = timer.get_stages()
        if retry_overhead_seconds > 0.1: current_stages['Retry_Overhead'] = round(retry_overhead_seconds, 4)

        recorded_sum = sum(current_stages.values())
        missing_time = total_latency - recorded_sum
        if missing_time > 0.5:
            current_stages['Untracked_Overhead'] = round(missing_time, 4)

        total_retrieved = len(self.last_retrieval_debug_info)
        citation_rate = cited_count / total_retrieved if total_retrieved > 0 else 0
        
        self.metrics = {
            "status": "success", "latency_seconds": round(total_latency, 2),
            "stage_durations": current_stages, 
            "active_params": params,
            "retrieval_latency": round(current_stages.get("Retrieval_Phase_Done", 0), 2),
            "generation_latency": round(current_stages.get("LLM_Generation", 0), 2),
            "total_retrieved": total_retrieved,
            "cited_count": cited_count, "citation_rate": round(citation_rate, 4),
            "top_rerank_score": round(final_docs[0].metadata.get('rerank_score', 0), 4) if final_docs else 0,
            "noise_ratio": round(1 - citation_rate, 4), "retry_triggered": (attempt > 0),
            "attempts_made": attempt + 1,
            "config_snapshot": params
        }
        self._log_interaction(question, full_response, self.metrics)