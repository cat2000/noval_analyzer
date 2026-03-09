import os
import json
import logging
import re
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ✅ Logger 配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✅ 自定义 SimpleEnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

_ensemble_logger = logging.getLogger("SimpleEnsembleRetriever")
if not _ensemble_logger.handlers:
    _ensemble_logger.setLevel(logging.INFO)

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
                docs = retriever.invoke(query)
                all_results.append(docs)
            except Exception as e:
                _ensemble_logger.warning(f"检索器 {retriever} 失败: {e}")
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

class RAGEngine:
    def __init__(self, 
                 txt_path: str, 
                 db_path: str = "./chroma_db", 
                 checkpoint_path: str = "./checkpoint.json",
                 model_name: str = "qwen3", 
                 eval_model_name: str = "qwen3:0.6b",
                 embed_model_name: str = "bge-large-zh"):
        
        global logger
        if 'logger' not in globals() or logger is None:
            logger = logging.getLogger(__name__)

        self.txt_path = txt_path
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        
        self.vector_store = None
        self.full_text = ""
        self._data_loaded_attempted = False 
        self.last_retrieval_debug_info = []
        self.hybrid_retriever = None
        
        # CRAG 配置
        self.crag_threshold = 0.5
        self.max_crag_retries = 1
        self.eval_concurrency = 10

        logger.info(f"正在加载 Embedding 模型: {self.embed_model_name} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        # 1. 主 LLM
        logger.info(f"正在加载主生成模型: {self.model_name} ...")
        self.llm = OllamaLLM(model=self.model_name, temperature=0.1, request_timeout=120)
        
        # 2. 评估 LLM
        self.eval_llm = self.llm
        if eval_model_name and eval_model_name != model_name:
            try:
                logger.info(f"正在加载评估专用小模型: {eval_model_name} (仅用于评分参考，不过滤)...")
                self.eval_llm = OllamaLLM(model=eval_model_name, temperature=0, request_timeout=60)
            except Exception as e:
                logger.warning(f"加载评估模型失败 ({e})，将降级使用主模型进行评估。")
        
        # 🎯 切片配置
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=300, 
            separators=["\n\n", "\n", "。", "！", "？", "；", "……", " ", ""]
        )
        self.chapter_pattern = re.compile(r'第\s*(\d+|[一二三四五六七八九十百]+)\s*章')

    def load_data(self):
        if self._data_loaded_attempted and self.vector_store is not None:
            return
        self._data_loaded_attempted = True
        if not self.full_text:
            self._load_text_content()
            
        logger.info("正在初始化向量数据库连接...")
        try:
            self.vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            current_docs_count = self.vector_store._collection.count()
            logger.info(f"当前向量库中已有 {current_docs_count} 个文档片段。")
            
            if current_docs_count > 0:
                if not self.hybrid_retriever:
                    self._init_hybrid_retriever()
                return

            logger.info("向量库为空，开始处理文本并生成向量索引...")
            self._process_and_embed()
            self._init_hybrid_retriever()
        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
            self.vector_store = None
            raise e

    def _init_hybrid_retriever(self):
        if not self.vector_store: return
        logger.info("正在构建混合检索器 (BM25 + Vector)...")
        try:
            all_docs_data = self.vector_store.get(include=["metadatas", "documents"])
            if not all_docs_data['documents']:
                logger.warning("向量库为空，无法初始化 BM25。")
                return

            docs_obj = [Document(page_content=c, metadata=m) for c, m in zip(all_docs_data['documents'], all_docs_data['metadatas'])]
            
            bm25_retriever = BM25Retriever.from_documents(docs_obj)
            bm25_retriever.k = 8
            
            vector_retriever = self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5}
            )
            
            # 权重调整：70% BM25 (精准关键词) + 30% Vector (语义)
            self.hybrid_retriever = SimpleEnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.7, 0.3], 
                k=10 
            )
            logger.info("✅ 混合检索器初始化完成。")
        except Exception as e:
            logger.error(f"混合检索器初始化失败: {e}, 将降级为纯向量检索。")
            self.hybrid_retriever = None

    def _load_text_content(self):
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
        raise ValueError(f"无法读取文件 {self.txt_path}。")

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("processed_chars", 0)
            except: 
                return 0
        return 0

    def _save_checkpoint(self, processed_chars: int):
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"processed_chars": processed_chars}, f)
        except: pass

    def _get_current_chapter(self, text_segment: str, last_chapter: str) -> str:
        matches = self.chapter_pattern.findall(text_segment)
        if matches: return f"第{matches[-1]}章"
        return last_chapter

    def _process_and_embed(self):
        start_index = self._load_checkpoint()
        if start_index > 0:
            logger.info(f"检测到断点，将从第 {start_index} 个字符处继续处理...")
        batch_size = 5000
        total_length = len(self.full_text)
        current_index = start_index
        current_chapter = "序言"
        
        while current_index < total_length:
            end_index = min(current_index + batch_size, total_length)
            chunk_text = self.full_text[current_index:end_index]
            current_chapter = self._get_current_chapter(chunk_text, current_chapter)
            docs = self.text_splitter.create_documents([chunk_text])
            for doc in docs:
                doc.metadata["source"] = os.path.basename(self.txt_path)
                doc.metadata["start_char"] = current_index
                doc.metadata["chapter"] = current_chapter
            
            if docs:
                logger.info(f"正在嵌入批次: {current_index}-{end_index} (章节:{current_chapter}, {len(docs)} 个片段)...")
                self.vector_store.add_documents(docs)
                self._save_checkpoint(end_index)
            current_index = end_index
            
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        logger.info("🎉 向量索引构建完成！")

    def _crag_evaluate_docs(self, question: str, docs: List[Document]) -> List[Document]:
        """
        【最终版】CRAG 评分 + 重排序。
        1. 对所有文档进行评分。
        2. 将评分高的文档排在前面（让大模型优先看到）。
        3. 返回所有文档（不过滤）。
        """
        if not docs:
            return []
        
        logger.info(f"🔍 [CRAG] 开始并行评估 {len(docs)} 个片段 (评分 + 重排序)...")
        
        lock = threading.Lock()
        clean_q = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', question)
        words = [w for w in clean_q.split() if len(w) > 1] 
        keywords_hint = ", ".join(words[:5]) if words else "核心概念"

        eval_prompt_template = (
            "任务：判断文档是否包含回答问题所需的**关键信息**。\n"
            "用户问题：{question}\n"
            "问题中的核心词提示：{keywords}\n"
            "待评估文档：{doc_content}\n\n"
            "只输出一个数字（1 代表高度相关，0 代表相关性低或无关）：\n"
            "1. 包含核心词、直接回答、定义解释或强相关语境 -> 输出 1\n"
            "2. 完全无关或仅是提及 -> 输出 0\n\n"
            "只输出数字 0 或 1。"
        )

        def evaluate_single_doc(doc):
            content_preview = doc.page_content[:800]
            prompt = eval_prompt_template.format(
                question=question, 
                keywords=keywords_hint,
                doc_content=content_preview
            )
            
            score = 0
            try:
                response = self.eval_llm.invoke(prompt)
                if '1' in response.strip():
                    score = 1
            except Exception as e:
                logger.warning(f"评估出错: {e}, 保守起见默认给 1 分 (保留)。")
                score = 1  # 改这里：从 0 改为 1
            
            with lock:
                doc.metadata['crag_score'] = score
            
            return doc

        # 并行评估
        with ThreadPoolExecutor(max_workers=self.eval_concurrency) as executor:
            futures = {executor.submit(evaluate_single_doc, doc): doc for doc in docs}
            for future in as_completed(futures):
                pass 
        
        # ✅ [关键动作] 重排序：高分在前，低分在后
        # 这样大模型在读取 Context 时，最先看到的是最相关的信息
        sorted_docs = sorted(docs, key=lambda d: d.metadata.get('crag_score', 0), reverse=True)
        
        relevant_count = sum(1 for d in sorted_docs if d.metadata.get('crag_score', 0) == 1)
        logger.info(f"✅ [CRAG] 评估完成: 共 {len(sorted_docs)} 个片段，{relevant_count} 个高相关 (已置顶)。")
        
        return sorted_docs

    def query(self, question: str, k: int = 8):
        """执行带有 CRAG 评分（不过滤）的混合检索增强生成"""
        if not self.vector_store:
            yield "❌ 错误：向量库未初始化。请先调用 load_data()。"
            return

        self.last_retrieval_debug_info = []
        
        # === 步骤 0: 查询重写 ===
        search_query = question
        if "天道" in question or "文化属性" in question or "强势文化" in question:
            search_query = (
                f"{question} "
                "透视社会依次有三个层面：技术、制度和文化 "
                "命运归根到底都是那种文化属性的产物 "
                "强势文化造就强者，弱势文化造就弱者 "
                "这是规律，也可以理解为天道，不以人的意志为转移 "
                "丁元英 文化属性 规律 救世主"
            )
            logger.info(f"🔄 [超级查询重写] 扩展搜索词：'{search_query}'")
        
        # 1. 检测章节约束
        chapter_match = re.search(r'第\s*(\d+|[一二三四五六七八九十百]+)\s*章', question)
        filter_dict = None
        use_hybrid = True

        if chapter_match:
            chapter_name = f"第{chapter_match.group(1)}章"
            logger.info(f"🎯 检测到章节约束: {chapter_name}")
            filter_dict = {"chapter": chapter_name}
            use_hybrid = False

        docs = []
        attempt = 0
        max_attempts = self.max_crag_retries + 1
        
        # ✅ [检索循环] - 移除了 CRAG 过滤导致的重试逻辑，因为现在不过滤了
        while attempt < max_attempts:
            try:
                if use_hybrid and self.hybrid_retriever and not filter_dict:
                    logger.info(f"🚀 [尝试 {attempt+1}] 使用混合检索 (BM25+Vector)...")
                    docs = self.hybrid_retriever.invoke(search_query)
                else:
                    logger.info(f"🎯 [尝试 {attempt+1}] 使用向量检索 (Filter: {filter_dict})...")
                    docs = self.vector_store.similarity_search(search_query, k=k, filter=filter_dict)
                
                if not docs and filter_dict and attempt == 0:
                    logger.warning("过滤后无结果，准备去掉过滤重试...")
                    filter_dict = None
                    attempt += 1
                    continue

                if docs:
                    # ✅ [关键修改] 调用 CRAG 仅用于评分和打标签，返回值是完整的 docs 列表
                    docs = self._crag_evaluate_docs(question, docs)
                    
                    # 移除了 "if not filtered_docs" 的保底逻辑，因为 docs 永远不为空（除非检索本身为空）
                    break
                else:
                    break

            except Exception as e:
                logger.warning(f"检索尝试 {attempt+1} 失败: {e}")
                attempt += 1

        if not docs:
            logger.critical("❌ 最终手段：检索彻底失败。")
            if use_hybrid and self.hybrid_retriever:
                docs = self.hybrid_retriever.invoke(question)[:3]
            else:
                docs = self.vector_store.similarity_search(question, k=3)
            
            if not docs:
                yield "⚠️ 经过多轮检索，仍未找到可靠依据。"
                return

        # 2. 计算相似度得分 & 构建 Context
        try:
            query_vec_original = self.embeddings.embed_query(question)
        except Exception:
            query_vec_original = None

        context_evidence = []
        
        for i, d in enumerate(docs):
            score = 0.0
            if query_vec_original:
                try:
                    doc_vec = self.embeddings.embed_documents([d.page_content])[0]
                    score = float(cosine_similarity([query_vec_original], [doc_vec])[0][0])
                except:
                    score = 0.0
            
            crag_score = d.metadata.get('crag_score', 0)
            
            # ✅ [关键动作] 根据 CRAG 分数添加显式标签
            if crag_score == 1:
                tag_label = "🔥[高相关]"
            else:
                tag_label = "❄️[参考]"
            
            debug_item = {
                "index": i + 1,
                "content": d.page_content,
                "score": round(score, 4),
                "source": d.metadata.get("source", "unknown"),
                "chapter": d.metadata.get("chapter", "未知"),
                "start_char": d.metadata.get("start_char", 0),
                "is_cited": False,
                "crag_score": crag_score
            }
            self.last_retrieval_debug_info.append(debug_item)
            
            # 构建带标签的上下文块
            evidence_block = (
                f"[依据 {i+1}] {tag_label} (章节:{d.metadata.get('chapter', '?')} | 相似度:{score:.2f}):\n"
                f"{d.page_content}"
            )
            context_evidence.append(evidence_block)
        
        context_text = "\n\n".join(context_evidence)
        
        # 3. 构建 Prompt (✅ 升级版：鼓励深度分析与多片段融合)
        system_instruction = """
# Role: 资深文化分析师
你的任务是基于【参考上下文】对用户问题进行**深度、全面**的分析，而不仅仅是简单的摘录。

## ⚠️ 上下文标签说明 (重要)
- 🔥[高相关]：**核心依据**。包含问题的直接答案、核心定义或关键论点。**必须优先引用**。
- ❄️[参考]：**辅助素材**。包含背景信息、侧面描写、对比案例、具体情节或补充细节。**请积极利用**它们来丰富你的分析，使回答更有深度。

## ✅ 核心策略：深度融合
1. **以 🔥 为核心**：首先使用 [高相关] 片段确立观点或直接回答问题。
2. **以 ❄️ 为血肉**：紧接着，**主动检索并引用** [参考] 片段中的细节、例子或背景，来解释“为什么”、“怎么做”或“具体表现”。
   - 如果 🔥 片段给出了定义，请用 ❄️ 片段中的**具体情节**来举例说明。
   - 如果 🔥 片段给出了结论，请用 ❄️ 片段中的**对比描述**来深化论证。
3. **禁止遗漏**：只要 [参考] 片段能提供额外的信息量（如人物心理、前因后果、具体对话），就**必须引用**，不要让有价值的信息被埋没。

## ⛔ 严格禁令
1. **禁止自由发挥**：所有分析必须严格基于提供的片段，不可使用外部知识。
2. **禁止无标引用**：每一句陈述事实、观点或进行分析的话，后面**必须**紧跟 `[依据 X]`。
3. **禁止过度简化**：不要只给出一句话的定义。用户需要的是**深入的解读**。

## ✅ 正确示范 (Few-Shot)
用户问题：什么是强势文化？
参考上下文：
[依据 1] 🔥[高相关]: 强势文化就是遵循事物规律的文化。
[依据 2] ❄️[参考]: 丁元英在古城隐居时，即使生活拮据也不愿变卖唱片，因为他遵循内心的规律。
[依据 3] ❄️[参考]: 弱势文化则是依赖强者的道德期望，破格获取。

错误回答 (过于简单): 强势文化就是遵循事物规律的文化 [依据 1]。
正确回答 (深度分析): 
强势文化的本质是遵循事物规律 [依据 1]。这种文化属性不仅仅是一个概念，更体现在具体的行为逻辑中。例如，丁元英在古城隐居时，即便生活陷入困境也坚持不违背自己的原则（如不变卖唱片），这正是他遵循内心规律、不被世俗道德绑架的体现 [依据 2]。与之相对，那种依赖强者道德期望、试图破格获取的思维方式，则属于弱势文化 [依据 3]。通过对比可以看出，强势文化造就强者，是因为它尊重客观规律而非主观幻想。

## 参考上下文 (共 {total_docs} 条)
{context}

## 用户问题
{question}

## 你的回答 (请模仿正确示范，结合 🔥 核心与 ❄️ 细节进行深度分析，每句话后标注 [依据 X])：
"""

        prompt_text = system_instruction.format(
            context=context_text, 
            question=question,
            total_docs=len(docs)
        )
        
        logger.info(f"📝 Prompt 构建完成，长度: {len(prompt_text)} 字符。送入 LLM...")
        
        # 4. 流式调用
        logger.info("[生成] 正在推演...")
        full_response = ""
        has_started = False
        
        try:
            yield "💡 正在结合原文推演...\n\n"
            
            stream_generator = self.llm.stream(prompt_text)
            
            iterator = iter(stream_generator)
            try:
                first_chunk = next(iterator)
                has_started = True
                if first_chunk:
                    full_response += first_chunk
                    yield first_chunk
                
                for chunk in iterator:
                    if chunk:
                        full_response += chunk
                        yield chunk
            except StopIteration:
                if not has_started:
                    logger.error("❌ LLM 返回了空流！")
                    yield "\n⚠️ 模型未生成任何内容。"
                    return

        except Exception as e:
            logger.error(f"LLM 推演失败: {e}")
            yield f"\n\n❌ **逻辑推演中断**: {str(e)}"
            return

        # 5. 后处理：解析引用
        logger.info(f"[调试] 模型原始回答预览: {full_response[:100]}...")
        
        cited_indices = set()
        pattern = r'[\[(](?:依据 | 参考)?\s*:?\s*(\d+)[\])]'
        
        matches = re.findall(pattern, full_response)
        logger.info(f"[调试] 正则匹配到的原始数字列表: {matches}")
        
        for m in matches:
            try: 
                idx = int(m)
                if 1 <= idx <= len(self.last_retrieval_debug_info):
                    cited_indices.add(idx)
            except Exception as e:
                logger.error(f"解析索引失败: {m}, 错误: {e}")
        
        for item in self.last_retrieval_debug_info:
            if item["index"] in cited_indices:
                item["is_cited"] = True
        
        if not cited_indices and len(self.last_retrieval_debug_info) > 0:
            logger.warning("⚠️ 警告：检索到了文档，但模型未在回答中标注任何 [依据 X] 标记！")
        
        logger.info(f"[统计] 检索总数:{len(self.last_retrieval_debug_info)}, 成功标记引用数:{len(cited_indices)}")