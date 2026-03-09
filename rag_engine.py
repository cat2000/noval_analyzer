import os
import json
import logging
import re
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain & AI libs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ✅ Logger 配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 自定义混合检索器 (支持 Rerank 输入)
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
# 2. RAG 引擎核心 (集成 Multi-Query, Rerank, Rewrite, Self-RAG)
# ==========================================
class RAGEngine:
    def __init__(self, 
                 txt_path: str, 
                 db_path: str = "./chroma_db", 
                 checkpoint_path: str = "./checkpoint.json",
                 model_name: str = "qwen3", 
                 eval_model_name: str = "qwen3:0.6b",
                 embed_model_name: str = "bge-large-zh",
                 rerank_model_name: str = "BAAI/bge-reranker-large"):
        
        global logger
        if 'logger' not in globals() or logger is None:
            logger = logging.getLogger(__name__)

        self.txt_path = txt_path
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        
        self.vector_store = None
        self.full_text = ""
        self._data_loaded_attempted = False 
        self.last_retrieval_debug_info = []
        self.hybrid_retriever = None
        
        # ⚙️ 配置参数优化
        # 由于加入了 Multi-Query (约 5 路查询)，单路检索数量建议从 10 降至 6，防止总文档数过多导致 Rerank 过慢
        self.top_k_initial = 6       
        self.top_k_final = 5         
        self.max_self_rag_attempts = 2 

        # 1. Embedding 模型
        logger.info(f"正在加载 Embedding 模型：{embed_model_name} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        # 2. Reranker 模型
        logger.info(f"正在加载 Reranker 模型：{rerank_model_name} ...")
        try:
            self.reranker = CrossEncoder(
                model_name=rerank_model_name,
                max_length=512,
                device='cpu' 
            )
            logger.info("✅ Reranker 加载成功。")
        except Exception as e:
            logger.error(f"❌ Reranker 加载失败：{e}，将降级为不使用 Rerank。")
            self.reranker = None

        # 3. 主 LLM
        logger.info(f"正在加载主模型：{model_name} ...")
        self.llm = OllamaLLM(model=self.model_name, temperature=0.3, request_timeout=120)
        
        # 4. 评估小模型
        self.eval_llm = self.llm
        if eval_model_name and eval_model_name != model_name:
            try:
                logger.info(f"正在加载评估小模型：{eval_model_name} ...")
                self.eval_llm = OllamaLLM(model=eval_model_name, temperature=0, request_timeout=60)
            except Exception as e:
                logger.warning(f"加载评估模型失败，降级使用主模型。")
        
        # 5. 切片配置
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
        logger.info("构建混合检索器 (BM25 + Vector)...")
        try:
            all_docs_data = self.vector_store.get(include=["metadatas", "documents"])
            if not all_docs_data['documents']:
                return

            docs_obj = [Document(page_content=c, metadata=m) for c, m in zip(all_docs_data['documents'], all_docs_data['metadatas'])]
            
            bm25_retriever = BM25Retriever.from_documents(docs_obj)
            bm25_retriever.k = self.top_k_initial
            
            vector_retriever = self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": self.top_k_initial, "fetch_k": 40, "lambda_mult": 0.5}
            )
            
            self.hybrid_retriever = SimpleEnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.7, 0.3], 
                k=self.top_k_initial 
            )
            logger.info("✅ 混合检索器就绪。")
        except Exception as e:
            logger.error(f"混合检索器初始化失败：{e}")
            self.hybrid_retriever = None

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

    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get("processed_chars", 0)
            except: return 0
        return 0

    def _save_checkpoint(self, processed_chars: int):
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"processed_chars": processed_chars}, f)
        except: pass

    def _get_current_chapter(self, text_segment: str, last_chapter: str) -> str:
        matches = self.chapter_pattern.findall(text_segment)
        return f"第{matches[-1]}章" if matches else last_chapter

    def _process_and_embed(self):
        start_index = self._load_checkpoint()
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
                doc.metadata.update({"source": os.path.basename(self.txt_path), "start_char": current_index, "chapter": current_chapter})
            
            if docs:
                logger.info(f"嵌入批次：{current_index}-{end_index} ({len(docs)} 片段)")
                self.vector_store.add_documents(docs)
                self._save_checkpoint(end_index)
            current_index = end_index
            
        if os.path.exists(self.checkpoint_path): os.remove(self.checkpoint_path)
        logger.info("🎉 索引构建完成！")


    # ==========================================
    # 🆕 功能 1: 语义化 Query Rewrite (深度聚焦)
    # ==========================================
    def _rewrite_query(self, question: str, history_context: str = "") -> str:
        if len(question) > 40 and any(k in question for k in ["具体情节", "原文片段", "第几章"]):
            return question

        examples = """
        示例 1:
        用户："什么是天道？"
        重写：《遥远的救世主》中丁元英对"天道"的定义，以及"天道"与"文化属性"、"强势文化"和"弱势文化"之间的逻辑关系。
        示例 2:
        用户："丁元英为什么那么厉害？"
        重写：丁元英在《遥远的救世主》中展现出的超凡认知能力来源，他对"文化属性"规律的掌握，以及他如何利用这些规律在商战和人性博弈中取胜的具体案例分析。
        """

        prompt = (
            f"你是一个精通《遥远的救世主》的资深研究助手。\n"
            f"任务：将用户问题重写为**语义丰富、包含关键实体**的检索查询。\n\n"
            f"## 参考范例:\n{examples}\n\n"
            f"## 当前任务:\n"
            f"用户问题：{question}\n"
            f"{f'之前尝试过的相关背景：{history_context}' if history_context else ''}\n\n"
            f"请先简要分析用户意图（一行），然后输出重写后的查询。\n"
            f"格式:\n"
            f"分析: ...\n"
            f"重写: ..."
        )

        try:
            response = self.llm.invoke(prompt).strip()
            rewritten_query = ""
            if "重写:" in response:
                rewritten_query = response.split("重写:", 1)[1].strip()
            elif "重写：" in response:
                rewritten_query = response.split("重写：", 1)[1].strip()
            else:
                lines = response.split('\n')
                rewritten_query = lines[-1].strip()
            
            rewritten_query = rewritten_query.strip('"\'')
            logger.info(f"🧠 [语义重写] 原问：'{question}' -> 新意：'{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.warning(f"语义重写失败，回退到原问题：{e}")
            return question

    # ==========================================
    # 🆕 功能 2: Multi-Query 生成 (广度覆盖)
    # ==========================================
    def _generate_multi_queries(self, question: str, n: int = 3) -> List[str]:
        """
        让 LLM 基于原问题生成 N 个不同角度的查询变体。
        策略：同义词替换、视角转换、具体化/抽象化。
        """
        prompt = (
            f"你是一个检索专家。请基于用户关于《遥远的救世主》的问题，生成 {n} 个不同角度的检索查询。\n"
            f"要求：\n"
            f"1. 每个查询侧重点不同（如：一个侧重哲学定义，一个侧重具体情节，一个侧重人物对话）。\n"
            f"2. 保持简洁，直接输出查询语句。\n"
            f"3. 必须包含原问题的核心意图。\n\n"
            f"用户问题：{question}\n\n"
            f"请直接输出 {n} 行查询，不要编号，不要额外解释："
        )
        
        try:
            response = self.llm.invoke(prompt).strip()
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # 去重并限制数量
            queries = list(dict.fromkeys(lines))[:n] 
            # 确保原问题也在候选池中（作为兜底）
            if question not in queries:
                queries.append(question)
                
            logger.info(f"🔍 [Multi-Query] 生成变体：{queries}")
            return queries
        except Exception as e:
            logger.warning(f"Multi-Query 生成失败，回退到原问题：{e}")
            return [question]

    # ==========================================
    # 🆕 功能 3: Rerank (重排序)
    # ==========================================
    def _rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        if not self.reranker or not docs:
            return docs
        
        logger.info(f"🔍 正在对 {len(docs)} 个片段进行 Rerank...")
        pairs = [[query, doc.page_content] for doc in docs]
        
        try:
            scores = self.reranker.predict(
                pairs, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                batch_size=32
            )
            
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)] * len(docs)
            
            if isinstance(scores, (int, float)):
                scores = [scores] * len(docs)
            elif hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            for i, doc in enumerate(docs):
                doc.metadata['rerank_score'] = scores[i]
            
            ranked_docs = sorted(docs, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)
            logger.info(f"✅ Rerank 完成。最高分：{ranked_docs[0].metadata['rerank_score']:.4f}")
            return ranked_docs[:self.top_k_final] 
        except Exception as e:
            logger.error(f"Rerank 出错：{e}, 返回原始顺序。")
            return docs

    # ==========================================
    # 🆕 功能 4: Self-RAG Logic (反思与迭代)
    # ==========================================
    def _evaluate_retrieval_quality(self, question: str, docs: List[Document]) -> bool:
        if not docs:
            return False
        
        top_doc = docs[0]
        top_score = top_doc.metadata.get('rerank_score', 0)
        
        if top_score > 0.85:
            logger.info(f"✅ Rerank 高分 ({top_score:.2f}) 直接通过评估。")
            return True
        
        if top_score < 0.5:
            logger.warning(f"⚠️ Rerank 分数 ({top_score:.2f}) 过低，直接判定失败。")
            return False
        
        logger.info(f"🤔 Rerank 分数 ({top_score:.2f}) 中等，启动小模型辅助评估...")
        
        prompt = (
            f"你是一个严格的评估助手。请判断给定的【文档片段】是否包含与【用户问题】相关的**实质性内容**。\n"
            f"注意：不需要片段直接给出完美定义，只要片段中**讨论、提及或解释**了问题中的核心概念，即可视为相关。\n"
            f"用户问题：{question}\n"
            f"文档片段 (Top 1): {top_doc.page_content[:300]}...\n"
            f"如果片段与问题相关（哪怕只是部分相关），输出 'YES'。\n"
            f"如果片段完全无关或只是在标题中提到但没有实质内容，输出 'NO'。\n"
            f"只输出 YES 或 NO。"
        )
        
        try:
            resp = self.eval_llm.invoke(prompt).strip().upper()
            logger.debug(f"评估模型响应：{resp}")
            
            if 'NO' in resp:
                logger.warning("⚠️ 评估模型判定不相关，触发重试。")
                return False
            else:
                logger.info("✅ 评估模型判定相关。")
                return True
                
        except Exception as e:
            logger.error(f"评估模型调用失败：{e}，保守起见返回 True。")
            return True 

    def query(self, question: str, k: int = 5):
        """
        主查询流程 (Multi-Query + Self-RAG 终极版)
        """
        if not self.vector_store:
            yield "❌ 错误：向量库未初始化。"
            return

        self.last_retrieval_debug_info = []
        attempt = 0
        final_docs = []
        
        # === Self-RAG 循环 ===
        while attempt <= self.max_self_rag_attempts:
            logger.info(f"🔄 [Self-RAG 尝试 {attempt+1}]")
            
            # 1. 确定当前查询列表 (Multi-Query 核心逻辑)
            current_queries = []
            if attempt == 0:
                # === 首次尝试：广度优先 (Multi-Query) ===
                # 策略：原问题 + 深度重写 + 3个多视角变体
                rewritten_deep = self._rewrite_query(question) 
                multi_vars = self._generate_multi_queries(question, n=3)
                # 合并并去重
                current_queries = list(dict.fromkeys([question, rewritten_deep] + multi_vars))
                logger.info(f"📢 [广度模式] 执行 {len(current_queries)} 路检索：{current_queries}")
            else:
                # === 重试模式：深度优先 (聚焦) ===
                # 策略：如果广度检索失败，说明噪声太多，此时需要极度聚焦
                hint = "之前的检索结果不够精准，请尝试从书中核心理论、人物对话或具体情节的角度重新表述查询。"
                deep_rewrite = self._rewrite_query(question, history_context=hint)
                current_queries = [deep_rewrite]
                logger.info(f"📢 [深度模式] 重试聚焦检索：{deep_rewrite}")

            # 2. 执行检索 (合并结果)
            all_docs = []
            seen_content = set() # 用于全局去重
            
            for q in current_queries:
                try:
                    if self.hybrid_retriever:
                        docs = self.hybrid_retriever.invoke(q)
                    else:
                        docs = self.vector_store.similarity_search(q, k=self.top_k_initial)
                    
                    # 去重：如果内容完全一样，就不重复添加了
                    for doc in docs:
                        if doc.page_content not in seen_content:
                            seen_content.add(doc.page_content)
                            all_docs.append(doc)
                except Exception as e:
                    logger.warning(f"查询 '{q}' 检索失败：{e}")

            if not all_docs:
                attempt += 1
                continue
                
            logger.info(f"📦 合并后去重文档总数：{len(all_docs)}")

            # 3. Rerank (对合并后的大池子进行重排序)
            # 注意：这里使用原问题 question 作为 Rerank 的基准，比用变体查询更稳定
            ranked_docs = self._rerank_docs(question, all_docs)
            
            # 4. 评估
            if self._evaluate_retrieval_quality(question, ranked_docs):
                final_docs = ranked_docs
                logger.info("✅ 检索质量评估通过。")
                break
            else:
                logger.info("❌ 检索质量评估未通过，准备重试...")
                attempt += 1
        
        if not final_docs:
            yield "⚠️ 经过多次检索与反思，仍未找到足够的原文依据来回答这个问题。"
            return

        # === 构建上下文 ===
        context_evidence = []
        for i, d in enumerate(final_docs):
            score = 0.0
            try:
                q_vec = self.embeddings.embed_query(question)
                d_vec = self.embeddings.embed_documents([d.page_content])[0]
                score = float(cosine_similarity([q_vec], [d_vec])[0][0])
            except: pass
            
            r_score = d.metadata.get('rerank_score', 0)
            tag_label = "🔥[核心依据]" if r_score > 0.7 else "❄️[辅助参考]"
            
            self.last_retrieval_debug_info.append({
                "index": i + 1,
                "content": d.page_content,
                "score": round(score, 4),
                "rerank_score": round(r_score, 4),
                "chapter": d.metadata.get("chapter", "?"),
                "is_cited": False
            })
            
            context_evidence.append(
                f"[依据 {i+1}] {tag_label} (章节:{d.metadata.get('chapter', '?')} | Rerank:{r_score:.2f}):\n{d.page_content}"
            )
        
        context_text = "\n\n".join(context_evidence)
        
        system_instruction_template = """
# Role: 资深文化分析师 (擅长结构化深度解读)
你的任务是基于【参考上下文】，对用户关于《遥远的救世主》的问题进行**深度、结构化**的分析。

## ⚠️ 核心指令 (必须严格执行)

1. **结构化输出格式 (最重要)**:
   - 你的回答必须是一篇**排版精美**的文章，严格遵循以下 Markdown 结构：
     - 使用 `###` 作为主标题 (概括核心观点)。
     - 使用 `####` 作为分论点标题。
     - 使用 **加粗** (`**文字**`) 强调关键概念和结论。
     - 使用 `-` 列表项来罗列具体论据或步骤。
     - 适当使用 `> 引用块` 来展示书中的原话。
   - **禁止**输出大段没有任何标点和分段的纯文本。

2. **密集且规范的引用 (格式严格)**:
   - **每一句话或每一个分论点后**，只要涉及原文内容，必须标注 `[依据 X]`。
   - **❗️重要格式要求**：在 `[依据 X]` 前面**必须加一个空格**，与前面的文字隔开。
   - ✅ 正确示例："...规律 [依据 1]"、"...工具 [依据 2]"。
   - ❌ 错误示例："...规律 [依据 1]" (缺少空格)。
   - 标签格式严格为：`[依据 1]`, `[依据 2]`。

3. **深度展开逻辑**:
   - 对于每个分论点，请按照 **"观点 -> 原文证据 -> 深度解析 -> 现实意义"** 的逻辑链条展开。
   - 确保内容详实，逻辑连贯，既有理论高度，又有情节支撑。

## 参考上下文
{context}

## 用户问题
{question}

## 你的回答 (请严格按照上述结构化格式撰写):
"""

        prompt_text = system_instruction_template.format(context=context_text, question=question)
        
        logger.info("📝 开始生成并自我反思...")
        full_response = ""
        
        try:
            yield "💡 正在深度推演...\n\n"
            stream_generator = self.llm.stream(prompt_text)
            for chunk in stream_generator:
                if chunk:
                    full_response += chunk
                    yield chunk
        except Exception as e:
            yield f"\n\n❌ 生成中断：{str(e)}"
            return

        # === 后处理统计 ===
        cited_indices = set()
        pattern = r'[\[(](?:依据 | 参考)?\s*:?\s*(\d+)[\])]'
        
        matches = re.findall(pattern, full_response)
        
        if matches:
            logger.debug(f"🔍 正则匹配到索引：{matches}")
        
        for m in matches:
            try: 
                idx = int(m)
                if 1 <= idx <= len(self.last_retrieval_debug_info):
                    cited_indices.add(idx)
            except ValueError:
                pass
        
        for item in self.last_retrieval_debug_info:
            if item["index"] in cited_indices:
                item["is_cited"] = True
        
        total_retrieved = len(self.last_retrieval_debug_info)
        cited_count = len(cited_indices)
        logger.info(f"[RAGAS 模拟] 检索数:{total_retrieved}, 引用数:{cited_count}, 覆盖率:{cited_count/total_retrieved if total_retrieved>0 else 0:.2%}")

        if not cited_indices and total_retrieved > 0:
            logger.warning("⚠️ 警告：模型未引用任何片段，可能存在幻觉风险！")