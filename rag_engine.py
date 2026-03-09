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
from sentence_transformers import CrossEncoder  # ✅ 改为直接导入 sentence-transformers
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
                logger.warning(f"检索器 {retriever} 失败: {e}")
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
# 2. RAG 引擎核心 (集成 Rerank, Rewrite, Self-RAG)
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
        
        # 配置参数
        self.top_k_initial = 10      # 初始检索数量 (为了给 Rerank 留余地)
        self.top_k_final = 5         # Rerank 后保留的数量
        self.max_self_rag_attempts = 2 # Self-RAG 最大重试次数

        # 1. Embedding 模型
        logger.info(f"正在加载 Embedding 模型: {embed_model_name} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        # 2. Reranker 模型 (新增)
        logger.info(f"正在加载 Reranker 模型: {rerank_model_name} ...")
        try:
            # ✅ 直接使用 sentence-transformers 的 CrossEncoder
            self.reranker = CrossEncoder(
                model_name=rerank_model_name,
                max_length=512,
                device='cpu' # 如果有 GPU 可改为 'cuda'
            )
            logger.info("✅ Reranker 加载成功。")
        except Exception as e:
            logger.error(f"❌ Reranker 加载失败: {e}，将降级为不使用 Rerank。")
            self.reranker = None

        # 3. 主 LLM (生成 + 重写 + 反思)
        logger.info(f"正在加载主模型: {model_name} ...")
        # ✅ [优化] 温度从 0.1 提升至 0.3，增加一点创造性，减少过度保守
        # 对于小模型，0.1 往往会导致它们不敢输出非原文内容
        self.llm = OllamaLLM(model=self.model_name, temperature=0.3, request_timeout=120)
        
        # 4. 评估小模型 (用于快速打分)
        self.eval_llm = self.llm
        if eval_model_name and eval_model_name != model_name:
            try:
                logger.info(f"正在加载评估小模型: {eval_model_name} ...")
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
            logger.info(f"向量库现有片段: {current_docs_count}")
            
            if current_docs_count > 0:
                if not self.hybrid_retriever:
                    self._init_hybrid_retriever()
                return

            logger.info("向量库为空，开始构建索引...")
            self._process_and_embed()
            self._init_hybrid_retriever()
        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
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
            logger.error(f"混合检索器初始化失败: {e}")
            self.hybrid_retriever = None

    def _load_text_content(self):
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"找不到书籍文件: {self.txt_path}")
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
                logger.info(f"嵌入批次: {current_index}-{end_index} ({len(docs)} 片段)")
                self.vector_store.add_documents(docs)
                self._save_checkpoint(end_index)
            current_index = end_index
            
        if os.path.exists(self.checkpoint_path): os.remove(self.checkpoint_path)
        logger.info("🎉 索引构建完成！")


    # ==========================================
    # 🆕 功能 1: 语义化 Query Rewrite (Semantic Rewrite)
    # ==========================================
    def _rewrite_query(self, question: str, history_context: str = "") -> str:
        """
        使用 LLM 进行语义层面的查询重写。
        策略：
        1. 意图识别：分析用户真正想问的核心概念（如：文化属性、强势弱势、天道规律）。
        2. 语境补全：自动关联书中的关键人物（丁元英、芮小丹）或核心理论。
        3. 示例引导 (Few-Shot)：通过书中具体案例教模型如何重写。
        """
        
        # ✅ 优化：如果问题本身已经非常具体且长，跳过重写，避免过度解读
        if len(question) > 40 and any(k in question for k in ["具体情节", "原文片段", "第几章"]):
            return question

        # ✅ Few-Shot Prompts: 给模型提供《遥远的救世主》特有的重写范例
        examples = """
        示例 1:
        用户: "什么是天道？"
        思考: 用户询问核心哲学概念。需关联丁元英的解释、"文化属性"、"强势/弱势文化"以及书中关于"神即道"的讨论。
        重写: 《遥远的救世主》中丁元英对"天道"的定义，以及"天道"与"文化属性"、"强势文化"和"弱势文化"之间的逻辑关系，包含书中关于"神即道，道法自然"的论述。

        示例 2:
        用户: "丁元英为什么那么厉害？"
        思考: 用户关注人物能力来源。需关联他的思维方式、对文化属性的洞察、以及在古城隐居时的行为逻辑。
        重写: 丁元英在《遥远的救世主》中展现出的超凡认知能力来源，他对"文化属性"规律的掌握，以及他如何利用这些规律在商战和人性博弈中取胜的具体案例分析。

        示例 3:
        用户: "芮小丹的死值得吗？"
        思考: 用户探讨人物命运和价值观。需关联芮小丹的"天国女儿"设定、她对爱情的理解、以及最后抓捕罪犯的动机。
        重写: 《遥远的救世主》中芮小丹牺牲的深层含义，结合她"天国女儿"的性格设定、对丁元英的爱情观，以及她在最后时刻选择履行职责而非苟活的价值观探讨。
        """

        prompt = (
            f"你是一个精通《遥远的救世主》(电视剧《天道》原著) 的资深研究助手。\n"
            f"你的任务是将用户的简短问题重写为一个**语义丰富、包含关键实体和理论背景**的检索查询，以便在向量库中精准定位原文片段。\n\n"
            f"## 重写原则:\n"
            f"1. **核心概念展开**: 如果问题涉及抽象概念（如天道、文化属性），必须展开其相关的理论定义和书中具体论述。\n"
            f"2. **人物与情境关联**: 自动关联相关的主要人物（丁元英、芮小丹、欧阳雪等）和关键情节。\n"
            f"3. **保留原意**: 不要改变用户问题的初衷，只是让它更适合搜索引擎。\n\n"
            f"## 参考范例:\n{examples}\n\n"
            f"## 当前任务:\n"
            f"用户问题: {question}\n"
            f"{f'之前尝试过的相关背景: {history_context}' if history_context else ''}\n\n"
            f"请先简要分析用户意图（一行），然后输出重写后的查询。\n"
            f"格式:\n"
            f"分析: ...\n"
            f"重写: ..."
        )

        try:
            response = self.llm.invoke(prompt).strip()
            
            # ✅ 解析输出：提取 "重写: " 之后的内容
            rewritten_query = ""
            if "重写:" in response:
                rewritten_query = response.split("重写:", 1)[1].strip()
            elif "重写：" in response: # 兼容中文冒号
                rewritten_query = response.split("重写：", 1)[1].strip()
            else:
                # 如果没有明确标记，尝试取最后一行作为查询
                lines = response.split('\n')
                rewritten_query = lines[-1].strip()
            
            # 清理可能的多余标点
            rewritten_query = rewritten_query.strip('"\'')
            
            logger.info(f"🧠 [语义重写] 原问: '{question}' -> 新意: '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.warning(f"语义重写失败，回退到原问题: {e}")
            return question

    # ==========================================
    # 🆕 功能 2: Rerank (重排序)
    # ==========================================
    def _rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        if not self.reranker or not docs:
            return docs
        
        logger.info(f"🔍 正在对 {len(docs)} 个片段进行 Rerank...")
        
        # ✅ 构造 [query, passage] 对的列表
        pairs = [[query, doc.page_content] for doc in docs]
        
        try:
            # ✅ 新版 sentence-transformers (v3+) 使用 predict 方法
            # convert_to_numpy=True 确保返回 numpy 数组，方便后续处理
            scores = self.reranker.predict(
                pairs, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                batch_size=32 # 批量处理加速
            )
            
            # 确保转换为 Python list，以便后续序列化或操作
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                # 极端情况兜底
                scores = [float(scores)] * len(docs)
            
            # 兼容性处理：某些版本可能返回 numpy 数组或单个值
            if isinstance(scores, (int, float)):
                scores = [scores] * len(docs)
            elif hasattr(scores, 'tolist'): # numpy array
                scores = scores.tolist()
            
            # 将分数存入 metadata 并排序
            for i, doc in enumerate(docs):
                doc.metadata['rerank_score'] = scores[i]
            
            # 降序排列
            ranked_docs = sorted(docs, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)
            logger.info(f"✅ Rerank 完成。最高分: {ranked_docs[0].metadata['rerank_score']:.4f}")
            return ranked_docs[:self.top_k_final] 
        except Exception as e:
            logger.error(f"Rerank 出错: {e}, 返回原始顺序。")
            return docs

    # ==========================================
    # 🆕 功能 3: Self-RAG Logic (反思与迭代)
    # ==========================================
    def _evaluate_retrieval_quality(self, question: str, docs: List[Document]) -> bool:
        """
        判断检索结果是否足够好。
        策略升级：
        1. 如果 Rerank 分数极高 (>0.85)，直接信任，跳过小模型评估（防止误杀）。
        2. 如果分数中等，再让小模型辅助判断。
        """
        if not docs:
            return False
        
        top_doc = docs[0]
        top_score = top_doc.metadata.get('rerank_score', 0)
        
        # ✅ 策略 1: 高分直接通过 (避免小模型过度敏感)
        if top_score > 0.85:
            logger.info(f"✅ Rerank 高分 ({top_score:.2f}) 直接通过评估。")
            return True
        
        # ✅ 策略 2: 低分直接拒绝
        if top_score < 0.5:
            logger.warning(f"⚠️ Rerank 分数 ({top_score:.2f}) 过低，直接判定失败。")
            return False
        
        # ✅ 策略 3: 中等分数 (0.5 - 0.85)，启用小模型辅助判断
        logger.info(f"🤔 Rerank 分数 ({top_score:.2f}) 中等，启动小模型辅助评估...")
        
        # 优化 Prompt：强调“相关讨论”也算合格，不仅仅是“直接定义”
        prompt = (
            f"你是一个严格的评估助手。请判断给定的【文档片段】是否包含与【用户问题】相关的**实质性内容**。\n"
            f"注意：不需要片段直接给出完美定义，只要片段中**讨论、提及或解释**了问题中的核心概念，即可视为相关。\n"
            f"用户问题: {question}\n"
            f"文档片段 (Top 1): {top_doc.page_content[:300]}...\n"
            f"如果片段与问题相关（哪怕只是部分相关），输出 'YES'。\n"
            f"如果片段完全无关或只是在标题中提到但没有实质内容，输出 'NO'。\n"
            f"只输出 YES 或 NO。"
        )
        
        try:
            resp = self.eval_llm.invoke(prompt).strip().upper()
            logger.debug(f"评估模型响应: {resp}")
            
            if 'NO' in resp:
                logger.warning("⚠️ 评估模型判定不相关，触发重试。")
                return False
            else:
                logger.info("✅ 评估模型判定相关。")
                return True
                
        except Exception as e:
            logger.error(f"评估模型调用失败: {e}，保守起见返回 True。")
            return True # 出错时默认通过，避免死循环

    def query(self, question: str, k: int = 5):
        """
        主查询流程 (终极修复版)：
        1. 统一 Prompt 策略，移除可能导致混淆的条件分支。
        2. 强制模型在内心建立“片段-观点”映射。
        """
        if not self.vector_store:
            yield "❌ 错误：向量库未初始化。"
            return

        self.last_retrieval_debug_info = []
        current_query = question
        attempt = 0
        
        final_docs = []
        
        # === Self-RAG 循环 ===
        while attempt <= self.max_self_rag_attempts:
            logger.info(f"🔄 [Self-RAG 尝试 {attempt+1}] 当前查询：{current_query}")
            
            # 1. 重写查询 (仅在重试时)
            if attempt > 0:
                hint = "之前的检索结果不够精准，请尝试从书中核心理论、人物对话或具体情节的角度重新表述查询。"
                current_query = self._rewrite_query(question, history_context=hint)
            
            # 2. 检索
            docs = []
            try:
                if self.hybrid_retriever:
                    docs = self.hybrid_retriever.invoke(current_query)
                else:
                    docs = self.vector_store.similarity_search(current_query, k=self.top_k_initial)
            except Exception as e:
                logger.error(f"检索失败：{e}")
            
            if not docs:
                attempt += 1
                continue
                
            # 3. Rerank
            ranked_docs = self._rerank_docs(current_query, docs)
            
            # 4. 评估 (使用 current_query)
            if self._evaluate_retrieval_quality(current_query, ranked_docs):
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
            # ✅ 统一标签逻辑，不再区分高低分，全部视为潜在依据
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
        
        # ✅ [结构化增强版] 强制模型输出层级分明的 Markdown
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
        
        # ✅ [调试] 打印 Prompt 开头，确认没有奇怪的前缀
        logger.debug(f"--- Prompt 预览 ---\n{prompt_text[:300]}...")
        
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
        
        # ✅ [超强兼容版正则] 
        # 1. 允许 [ 或 ( 开头
        # 2. 允许 "依据" 或 "参考" 可选
        # 3. 允许中间有冒号或空格 (甚至没有)
        # 4. 捕获数字
        # 5. 允许 ] 或 ) 结尾
        # 关键点：这个正则会匹配字符串中出现的任何符合模式的片段，无论前后是否有空格
        pattern = r'[\[(](?:依据 | 参考)?\s*:?\s*(\d+)[\])]'
        
        matches = re.findall(pattern, full_response)
        
        # 调试日志：打印匹配到的原始内容
        if matches:
            logger.debug(f"🔍 正则匹配到索引：{matches}")
        
        for m in matches:
            try: 
                idx = int(m)
                # 确保索引在有效范围内
                if 1 <= idx <= len(self.last_retrieval_debug_info):
                    cited_indices.add(idx)
            except ValueError:
                pass
        
        # 更新 debug 信息中的 is_cited 状态
        for item in self.last_retrieval_debug_info:
            if item["index"] in cited_indices:
                item["is_cited"] = True
        
        total_retrieved = len(self.last_retrieval_debug_info)
        cited_count = len(cited_indices)
        logger.info(f"[RAGAS 模拟] 检索数:{total_retrieved}, 引用数:{cited_count}, 覆盖率:{cited_count/total_retrieved if total_retrieved>0 else 0:.2%}")

        if not cited_indices and total_retrieved > 0:
            logger.warning("⚠️ 警告：模型未引用任何片段，可能存在幻觉风险！")
            logger.warning(f"模型回答预览：{full_response[:200]}")
            # 额外调试：打印完整响应以便排查
            # logger.debug(f"完整响应:\n{full_response}")