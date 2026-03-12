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
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Logger 配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalFileStore:
    """
    一个简单的本地文件存储实现，用于存储 Parent-Child 映射关系。
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def _get_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(self.path, f"{safe_key}.json")

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        results = []
        for key in keys:
            path = self._get_path(key)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results.append(data)
                except Exception:
                    results.append(None)
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: List[tuple]) -> None:
        for key, value in key_value_pairs:
            path = self._get_path(key)
            try:
                data_to_save = value
                if hasattr(value, 'page_content'):
                    data_to_save = {
                        "page_content": value.page_content,
                        "metadata": value.metadata,
                        "type": "Document"
                    }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False)
            except Exception as e:
                logger.error(f"保存文档到本地存储失败 {key}: {e}")

    def mdelete(self, keys: List[str]) -> None:
        for key in keys:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        if not os.path.exists(self.path):
            return
        for filename in os.listdir(self.path):
            if filename.endswith(".json"):
                key = filename[:-5]
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
# 流式生成收集器
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
        self.question_for_dedup = question
        try:
            stream_generator = self.llm.stream(self.prompt_text)
            
            for chunk in stream_generator:
                if not chunk:
                    continue
                
                self.full_response += chunk
                
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
# 核心引擎：RAGEngine (✅ 并行混合检索版)
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
        
        # ✅ 修改：分别存储两个检索器，以便并行调用
        self.bm25_retriever = None
        self.vector_retriever = None 
        
        self.metrics = {}
        
        self.default_params = {
            "top_k_initial": 6,      
            "top_k_final": 2,        # 用户设定的基准值
            "max_self_rag_attempts": 1,
            "multi_query_count": 1,     
            "rerank_threshold": 0.5,
            "eval_top_k": 3,
            # ✅ 新增：动态窗口配置
            "dynamic_window_enabled": True, # 是否开启智能微调
            "min_context_size": 1,          # 绝对最小值
            "max_context_size": 6,          # 绝对最大值 (防止爆显存)
            "high_confidence_threshold": 0.75, # 高分阈值：超过此分可扩充
            "low_confidence_threshold": 0.45   # 低分阈值：低于此分不扩充
        }

        self._load_models(embed_model_name, rerank_model_name, model_name, eval_model_name)

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", "。", "！", "？"]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=50, 
            separators=["\n\n", "\n", "。", "！", "？"]
        )

        self.doc_store = LocalFileStore("./doc_store")
        self.parent_child_index_built = False
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
                if not self.bm25_retriever or not self.vector_retriever:
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
        
        logger.info("🚀 构建并行混合检索器 (BM25 + Vector Parent-Child)...")
        try:
            # 1. 获取所有子文档用于构建 BM25
            all_data = self.vector_store.get(include=["metadatas", "documents"])
            if not all_data['documents']:
                logger.warning("向量库为空，无法构建检索器。")
                return

            sub_docs = [Document(page_content=c, metadata=m) for c, m in zip(all_data['documents'], all_data['metadatas'])]
            
            # 2. ✅ 创建 BM25 检索器 (基于子文档)
            logger.info("初始化 BM25 检索器...")
            self.bm25_retriever = BM25Retriever.from_documents(sub_docs)
            self.bm25_retriever.k = self.default_params["top_k_initial"] * 2 # BM25 多召回一些
            
            # 3. ✅ 创建自定义的向量检索器 (带父子映射)
            logger.info("初始化向量检索器 (Parent-Child Wrapper)...")
            
            vector_store_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.default_params["top_k_initial"]}
            )
            
            class ParentChildWrapper(BaseRetriever):
                vector_retriever: BaseRetriever
                doc_store: Any
                k: int = 8
                
                def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                    sub_docs = self.vector_retriever.invoke(query)
                    parent_ids = set()
                    for doc in sub_docs:
                        pid = doc.metadata.get("parent_id")
                        if pid:
                            parent_ids.add(pid)
                    
                    parent_docs = []
                    seen_parents = set()
                    for pid in parent_ids:
                        if pid not in seen_parents:
                            raw_data = self.doc_store.mget([pid])[0]
                            if raw_data:
                                if isinstance(raw_data, dict) and "page_content" in raw_data:
                                    p_doc = Document(
                                        page_content=raw_data.get("page_content", ""),
                                        metadata=raw_data.get("metadata", {})
                                    )
                                else:
                                    p_doc = raw_data
                                parent_docs.append(p_doc)
                                seen_parents.add(pid)
                    
                    if not parent_docs:
                        return sub_docs[:self.k]
                    return parent_docs[:self.k]

            self.vector_retriever = ParentChildWrapper(
                vector_retriever=vector_store_retriever,
                doc_store=self.doc_store,
                k=self.default_params["top_k_initial"]
            )
            
            logger.info("✅ 并行混合检索器就绪：BM25 (关键词) + Vector (语义/父子映射)。")
            
        except Exception as e:
            logger.error(f"检索器初始化失败：{e}")
            # 降级方案
            self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.default_params["top_k_initial"]})
            self.bm25_retriever = None

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
        logger.info("🚀 开始构建父子文档索引与摘要元数据...")
        parent_docs = self.parent_splitter.create_documents([self.full_text])
        total_parents = len(parent_docs)
        logger.info(f"共规划 {total_parents} 个父文档片段。")

        start_index = self._load_checkpoint()
        if start_index >= 0:
            logger.info(f"📍 检测到断点：上次成功处理到索引 {start_index}。")
            current_count = self.vector_store._collection.count()
            if current_count == 0:
                logger.warning("⚠️ 警告：Checkpoint 存在但向量库为空！将重新构建。")
                start_index = -1
        else:
            logger.info("🆕 未检测到有效断点，将从头开始构建。")
            start_index = -1

        docs_to_embed = []
        
        for i, p_doc in enumerate(parent_docs):
            if i <= start_index:
                continue

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
                summary = "生成失败：无法连接模型或超时"
            
            p_doc.metadata.update({
                "source": os.path.basename(self.txt_path),
                "summary": summary,
                "parent_id": f"parent_{i}",
                "is_parent": True
            })
            
            current_chapter = self._get_current_chapter(p_doc.page_content, "未知章节")
            p_doc.metadata["chapter"] = current_chapter

            try:
                self.doc_store.mset([(f"parent_{i}", p_doc)])
            except Exception as e:
                logger.error(f"保存父文档到本地存储失败：{e}")

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
            
            if docs_to_embed:
                try:
                    self.vector_store.add_documents(docs_to_embed)
                    docs_to_embed = []
                except Exception as e:
                    logger.error(f"嵌入向量失败：{e}")
                    continue

            self._save_checkpoint(i)
            
            if (i + 1) % 10 == 0 or i == total_parents - 1:
                logger.info(f"🎉 阶段性完成：已处理 {i+1}/{total_parents} 父文档。")

        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            logger.info("🧹 所有文档处理完毕，已清除 Checkpoint 文件。")
        
        self.parent_child_index_built = True
        logger.info("🎉 父子文档索引与摘要元数据构建完成！")

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "last_processed_index" in data:
                        return data["last_processed_index"]
                    logger.warning("Checkpoint 格式过旧，将重新构建索引。")
                    return -1
            except Exception as e:
                logger.error(f"读取 Checkpoint 失败：{e}，将重新构建。")
                return -1
        return -1

    def _save_checkpoint(self, index: int):
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"last_processed_index": index}, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存 Checkpoint 失败：{e}")

    def _get_current_chapter(self, text_segment: str, last_chapter: str) -> str:
        if not text_segment:
            return last_chapter

        pattern = re.compile(
            r'(?:第\s*([0-90-9一二三四五六七八九十百千]+)\s*[章节])|'
            r'(?:Chapter\s*([0-9]+))|'
            r'(?:([一二三四五六七八九十百千]+)\s*章)'
        )
        
        search_text = text_segment[:600] 
        matches = pattern.findall(search_text)
        
        if matches:
            first_match = matches[0]
            chapter_num = ""
            for group in first_match:
                if group:
                    chapter_num = group
                    break
            if chapter_num:
                return f"第{chapter_num}章"
        
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
        

    def _determine_dynamic_k(self, question: str, ranked_docs: List[Document], base_k: int) -> int:
        """
        ✅ 动态决定送入 LLM 的上下文窗口大小
        策略：以 base_k (用户设定值) 为基准，根据问题复杂度和分数分布进行 ±2 的微调。
        """
        if not ranked_docs:
            return 0
            
        params = self.default_params
        
        # 如果开关关闭，直接返回用户设定值
        if not params.get("dynamic_window_enabled", True):
            return base_k

        min_k = params.get("min_context_size", 1)
        max_k = params.get("max_context_size", 6)
        high_thresh = params.get("high_confidence_threshold", 0.75)
        low_thresh = params.get("low_confidence_threshold", 0.45)

        # --- 因素 1: 问题复杂度分析 ---
        complexity_score = 0
        q_len = len(question)
        complex_keywords = ["为什么", "如何", "分析", "比较", "异同", "详细", "具体情节", "深层", "逻辑", "关系"]
        
        if q_len > 40:
            complexity_score += 1
        if any(kw in question for kw in complex_keywords):
            complexity_score += 1
            
        is_complex_question = (complexity_score >= 1)
        # logger.debug(f"🧠 [复杂度] 长度:{q_len}, 关键词:{complexity_score} -> {'复杂' if is_complex_question else '简单'}")

        # --- 因素 2: 分数分布分析 ---
        # 获取当前 base_k 位置的分数 (注意索引从 0 开始，需判断长度)
        score_at_base_k = ranked_docs[base_k - 1].metadata.get('rerank_score', 0) if len(ranked_docs) >= base_k else 0
        score_at_1 = ranked_docs[0].metadata.get('rerank_score', 0) if len(ranked_docs) > 0 else 0
        
        # logger.debug(f"📊 [分数] Top1:{score_at_1:.2f}, Top{base_k}:{score_at_base_k:.2f}")

        dynamic_k = base_k

        # 策略 A: 分数骤降 -> 收缩 (避免噪音)
        # 如果第一个文档分数就很低，说明检索质量差，只保留 1 个或 base_k-1
        if score_at_1 < low_thresh:
            dynamic_k = max(min_k, base_k - 1)
            # logger.info(f"⚠️ [动态调整] 最高分过低 ({score_at_1:.2f})，收缩窗口至 {dynamic_k}。")
            
        # 策略 B: 高分密集 + 问题复杂 -> 扩张
        # 如果第 base_k 个分数依然很高，且问题复杂，大胆向后多看几个
        elif score_at_base_k > high_thresh:
            if is_complex_question:
                dynamic_k = min(max_k, base_k + 2) # 复杂问题多看 2 个
                # logger.info(f"✅ [动态调整] 高分密集且问题复杂，扩大窗口至 {dynamic_k}。")
            else:
                dynamic_k = min(max_k, base_k + 1) # 简单问题多看 1 个
                # logger.info(f"ℹ️ [动态调整] 高分密集但问题简单，微调窗口至 {dynamic_k}。")
        
        # 策略 C: 分数中等 -> 仅当问题复杂时微调
        else:
            if is_complex_question and len(ranked_docs) > base_k:
                dynamic_k = min(max_k, base_k + 1)
                # logger.info(f"ℹ️ [动态调整] 问题复杂，适度增加 1 个上下文。")

        # 确保不超过实际文档总数
        final_k = min(dynamic_k, len(ranked_docs))
        
        if final_k != base_k:
            logger.info(f"🎯 [动态窗口] 基准:{base_k} -> 最终:{final_k} (原因：{'复杂问题/高分' if final_k > base_k else '低分降噪'})")
        
        return final_k        

    def _build_context_and_metrics(self, final_docs, question, timer):
        context_evidence = []
        timer.checkpoint("Context_Embedding_Start") 
        
        try:
            q_vec = self.embeddings.embed_query(question)
            doc_contents = [d.page_content for d in final_docs]
            d_vecs = self.embeddings.embed_documents(doc_contents)
            similarities = cosine_similarity([q_vec], d_vecs)[0]
        except Exception as e:
            logger.warning(f"批量嵌入计算失败：{e}")
            similarities = [0.0] * len(final_docs)
        
        for i, d in enumerate(final_docs):
            score = similarities[i] if i < len(similarities) else 0.0
            r_score = d.metadata.get('rerank_score', 0)
            tag_label = "🔥[核心依据]" if r_score > 0.7 else "❄️[辅助参考]"
        
            summary = d.metadata.get('summary', '无摘要')
            chapter = d.metadata.get('chapter', '?')
            
            self.last_retrieval_debug_info.append({
                "index": i + 1, 
                "content": d.page_content, 
                "summary": summary,
                "score": round(score, 4),
                "rerank_score": round(d.metadata.get('rerank_score', 0), 4), 
                "chapter": chapter, 
                "is_cited": False
            })
            
            context_segment = (
                f"[依据 {i+1}] {tag_label} (章节:{chapter})\n"
                f"> 📝 **核心摘要**: {summary}\n"
                f"{d.page_content}"
            )
            context_evidence.append(context_segment)
        
        timer.checkpoint("Context_Building_Done") 
        return "\n\n---\n\n".join(context_evidence)

    def _generate_response_stream(self, context_text, question):
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
        collector = StreamingResponseCollector(self.llm, prompt_text)
        return collector, collector.generate(question)

    def _evaluate_retrieval_quality(self, question: str, docs: List[Document]) -> bool:
        if not docs:
            return False
        
        eval_k = self.default_params.get("eval_top_k", 3)
        k_to_eval = min(len(docs), eval_k)
        docs_to_check = docs[:k_to_eval]
        
        scores = [d.metadata.get('rerank_score', 0) for d in docs_to_check]
        
        if not scores:
            return False
            
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        logger.info(f"📊 [评估] 前 {k_to_eval} 个文档 - 平均分：{avg_score:.2f}, 最高分：{max_score:.2f}")

        if avg_score > 0.75:
            logger.info(f"✅ [快模式] 平均分 ({avg_score:.2f}) > 0.75，直接判定检索成功。")
            return True
        
        if max_score < 0.45:
            logger.warning(f"⚠️ [快模式] 最高分 ({max_score:.2f}) < 0.45，直接判定检索失败。")
            return False
        
        logger.info(f"🤔 [慢模式] 分数处于中间地带 ({avg_score:.2f})，启动 LLM 辅助评估前 {k_to_eval} 个文档...")
        
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
        def count_citations(text: str):
            indices = set()
            pattern_strict = r'[\[\(【](?:依据 | 参考 | 见依据 | 见参考)?\s*:?\s*(\d+)[\]\)】]'
            matches_strict = re.findall(pattern_strict, text)
            
            for m in matches_strict:
                try: 
                    idx = int(m)
                    if 1 <= idx <= len(self.last_retrieval_debug_info):
                        indices.add(idx)
                except ValueError: 
                    pass
            
            if len(indices) == 0 and len(self.last_retrieval_debug_info) > 0:
                logger.info("⚠️ 未检测到严格依据编号，尝试通过章节号进行模糊匹配...")
                chapter_pattern = r'——\s*\[第\s*([0-90-9一二三四五六七八九十百]+)\s*章\]'
                chapter_matches = re.findall(chapter_pattern, text)
                
                if chapter_matches:
                    for item in self.last_retrieval_debug_info:
                        item_chapter = item.get('chapter', '')
                        for q_chapter_num in chapter_matches:
                            if str(q_chapter_num) in item_chapter or item_chapter in f"第{q_chapter_num}章":
                                indices.add(item["index"])
                                logger.debug(f"模糊匹配成功：依据 {item['index']} ({item_chapter}) <-> 文中提到 [第{q_chapter_num}章]")
            
            return indices, len(indices)

        cited_indices, cited_count = count_citations(full_response)
        needs_retry = (cited_count == 0 and len(self.last_retrieval_debug_info) > 0)
        
        if needs_retry:
            logger.warning("⚠️ [自检警告] 生成结果完全缺少引用标识 (含章节号)。添加系统注记。")
            full_response += "\n\n> ⚠️ **系统注**: 本次回答未能自动标注原文引用或章节出处，请人工核对上下文。"
        
        for item in self.last_retrieval_debug_info:
            if item["index"] in cited_indices:
                item["is_cited"] = True
            else:
                item["is_cited"] = False
        
        return full_response, cited_count, cited_indices, needs_retry

    def query(self, question: str, **kwargs):
        params = {**self.default_params, **kwargs}
        rewrite_mode = params.get('rewrite_mode', 'deep')
        top_k_initial = params['top_k_initial']
        top_k_final = params['top_k_final']
        multi_query_count = params['multi_query_count']
        max_attempts = params.get('max_self_rag_attempts', 0)

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

        while attempt <= max_attempts and not success_eval:
            is_retry = (attempt > 0)
            logger.info(f"🔄 [尝试 {attempt + 1}/{max_attempts + 1}] {'(降级重试)' if is_retry else '(初始检索)'}")
            
            try:
                # 1. 准备查询列表
                if is_retry:
                    current_queries = [question]
                    logger.info("⚡ [降级策略] 重试模式：禁用多路查询，直接使用原问题。")
                else:
                    rewritten_query = self._rewrite_query(question, mode=rewrite_mode) 
                    timer.checkpoint("Query_Rewrite")
                    
                    if rewrite_mode != 'direct' and multi_query_count > 0:
                        multi_vars = self._generate_multi_queries_parallel(question, n=multi_query_count)
                        timer.checkpoint("MultiQuery_Gen")
                        current_queries = list(dict.fromkeys([question, rewritten_query] + multi_vars))
                    else:
                        current_queries = [rewritten_query if rewritten_query else question]

                # 2. ✅ 执行串行混合检索 (修复版)
                all_docs = []
                seen_content = set()
                
                # 遍历每个查询变体 (串行，避免 GIL 竞争)
                for q in current_queries:
                    # --- A. 向量检索 (主力) ---
                    if self.vector_retriever:
                        try:
                            v_docs = self.vector_retriever.invoke(q)
                            for doc in v_docs:
                                if doc.page_content not in seen_content:
                                    seen_content.add(doc.page_content)
                                    all_docs.append(doc)
                        except Exception as e:
                            logger.warning(f"Vector 检索失败 ({q[:20]}...): {e}")

                    # --- B. BM25 检索 (补充，严格限流) ---
                    if self.bm25_retriever:
                        try:
                            # ⚡ 关键优化：BM25 仅召回 5 个，避免全量扫描拖慢速度
                            old_k = self.bm25_retriever.k
                            self.bm25_retriever.k = min(5, top_k_initial) 
                            
                            b_docs = self.bm25_retriever.invoke(q)
                            
                            # 恢复原值
                            self.bm25_retriever.k = old_k 
                            
                            for doc in b_docs:
                                if doc.page_content not in seen_content:
                                    seen_content.add(doc.page_content)
                                    all_docs.append(doc)
                        except Exception as e:
                            logger.warning(f"BM25 检索失败 ({q[:20]}...): {e}")

                timer.checkpoint("Parallel_Retrieval") # 保持原名以兼容前端监控
                logger.info(f"📦 混合检索完成，去重后共 {len(all_docs)} 个候选文档。")

                # 3. Rerank
                if all_docs:
                    ranked_docs = self._rerank_docs(question, all_docs, top_k_final)
                    timer.checkpoint("Rerank")
                    
                    # ✅ 动态决定最终送入 LLM 的文档数量
                    final_k = self._determine_dynamic_k(question, ranked_docs, top_k_final)
                    final_docs = ranked_docs[:final_k]
                    
                    if final_k != top_k_final:
                        logger.info(f"📦 动态调整上下文：用户设定 {top_k_final} -> 实际送入 {final_k} 个片段。")
                else:
                    final_docs = []
                
                # 4. 评估
                if final_docs:
                    success_eval = self._evaluate_retrieval_quality(question, final_docs)
                    timer.checkpoint("Eval_Phase")
                    
                    if not success_eval and attempt < max_attempts:
                        logger.warning("❌ 评估未通过，准备重试...")
                        attempt += 1
                        # 计算重试开销
                        current_total = timer.get_total_time()
                        sum_stages = sum(timer.get_stages().values())
                        retry_overhead_seconds += (current_total - sum_stages)
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
                logger.error(f"检索阶段发生异常：{e}", exc_info=True)
                if attempt < max_attempts:
                    attempt += 1
                    continue
                final_docs = []
                break
            
            # 正常结束循环
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

        context_text = self._build_context_and_metrics(final_docs, question, timer)
        
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
        
        full_response, cited_count, cited_indices, needs_retry = self._post_process_response(full_response, context_text, question, timer)
        
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