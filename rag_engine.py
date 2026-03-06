import os
import json
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings, OllamaLLM  <-- 可以注释掉 OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ 新增导入
from langchain_core.documents import Document
import hashlib

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
                 embed_model_name: str = "bge-large-zh"): # 默认值改为 bge-large-zh
        
        self.txt_path = txt_path
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        
        # ✅ 修改开始：使用 HuggingFace 加载 BGE 模型
        logger.info(f"正在加载 Embedding 模型: {self.embed_model_name} ...")
        
        # 为了加速国内下载，可选：设置镜像环境变量 (如果在代码外没设置)
        # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh",       # 指定中文大模型
            model_kwargs={'device': 'cpu'},       # 如果有 GPU 改为 'cuda'
            encode_kwargs={
                'normalize_embeddings': True,     # ✅ 关键：必须归一化以计算余弦相似度
                'batch_size': 32
            }
        )

        self.llm = OllamaLLM(model=self.model_name, temperature=0.7)
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,          # 增大到 800，容纳更多上下文
            chunk_overlap=200,       # 增大重叠到 200，防止关键信息被切断
            length_function=len,
            # 增加中文特有的分隔符优先级，尽量保持句子完整
            separators=[
                "\n\n",              # 段落
                "\n",                # 换行
                "。", "！", "？",     # 句子结束
                "；",                # 分句
                "，",                # 逗号
                " ",                 # 空格
                ""                   # 字符
            ]
        )
        
        self.vector_store = None
        self.full_text = ""
        # 标记是否已尝试加载
        self._data_loaded_attempted = False 
        
    def load_data(self):
        """加载数据：读取TXT -> 切分 -> 存入向量库 (支持断点续传)"""
        if self._data_loaded_attempted and self.vector_store is not None:
            # 如果已经加载过且成功，直接返回
            return

        self._data_loaded_attempted = True
        
        # 1. 读取文本
        if not self.full_text:
            self._load_text_content()
            
        # 2. 加载或创建向量库
        logger.info("正在初始化向量数据库连接...")
        try:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            
            current_docs_count = self.vector_store._collection.count()
            logger.info(f"当前向量库中已有 {current_docs_count} 个文档片段。")
            
            # 如果库里有数据，直接返回 (连接已建立)
            if current_docs_count > 0:
                logger.info("检测到已有向量数据，连接成功。")
                return

            # 3. 如果没有数据，开始处理文本
            logger.info("向量库为空，开始处理文本并生成向量索引...")
            self._process_and_embed()
            
        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
            self.vector_store = None # 确保出错时置空
            raise e
        
    def _load_text_content(self):
        """读取 TXT 文件，处理编码"""
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"找不到书籍文件: {self.txt_path}")
        
        logger.info(f"正在读取书籍: {os.path.basename(self.txt_path)} ...")
        
        # 尝试编码列表
        encodings_to_try = ['utf-16', 'utf-8', 'gbk']
        
        for encoding in encodings_to_try:
            try:
                with open(self.txt_path, 'r', encoding=encoding) as f:
                    self.full_text = f.read()
                logger.info(f"✅ 成功通过 {encoding} 读取文件。")
                return # 成功则退出
            except UnicodeDecodeError:
                continue # 尝试下一个编码
            except Exception as e:
                logger.warning(f"编码 {encoding} 尝试失败: {e}")
                continue
                
        raise ValueError(f"无法读取文件 {self.txt_path}，请检查文件编码。")

    # ... (_load_checkpoint, _save_checkpoint 保持不变) ...
    def _load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("processed_chars", 0)
            except Exception as e:
                logger.warning(f"⚠️ 断点文件损坏 ({e})，将重置。")
                try:
                    os.remove(self.checkpoint_path)
                except:
                    pass
                return 0
        return 0

    def _save_checkpoint(self, processed_chars: int):
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"processed_chars": processed_chars}, f)
        except Exception as e:
            logger.error(f"保存断点失败: {e}")

    def _process_and_embed(self):
        # ... (保持不变，但在最后确保 vector_store 不为空) ...
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
                doc.metadata["start_char"] = current_index + i
            
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

    
    def query(self, question: str, k: int = 4):

        """执行检索增强生成 (RAG) - 重构后的强力 Prompt 版本"""

         # 【新增】查询重写 (Query Rewriting) - 让问题更具体，利于检索
        # 只有当问题较短时才触发，或者始终触发以优化语义
        rewrite_prompt = f"""
        你是一个检索助手。用户将询问关于《遥远的救世主》的问题。
        请将用户的问题改写为一个包含更多关键词、更具体、更适合向量检索的陈述句。
        不要回答问题，只输出改写后的句子。
        
        用户问题: {question}
        改写后的检索 query:
        """
        
        # 这里为了简单，同步调用一次 llm.invoke (非流式)，仅用于生成检索词
        # 注意：这会增加一点延迟，但能显著提升检索质量
        try:
            optimized_query = self.llm.invoke(rewrite_prompt).strip()
            logger.info(f"[查询重写] 原始: '{question}' -> 优化: '{optimized_query}'")
            # 使用优化后的 query 进行检索
            search_query = optimized_query
        except:
            search_query = question # 失败则用原问题

        # --- 初始化检查 ---
        if not self.vector_store:
            logger.warning("检测到向量库未初始化，尝试自动加载...")
            try:
                self.load_data()
            except Exception as e:
                yield f"❌ 错误：向量库初始化失败 ({str(e)})。"
                return
        
        if not self.vector_store:
            yield "❌ 错误：向量库仍未初始化。"
            return

        # 1. 检索相关片段 (优化版)
        logger.info(f"[规则引擎] 正在检索与 '{question}' 相关的客观事实...")
        
        # 方案 A: 增加 k 值，给模型更多素材去筛选
        # 方案 B: 使用 similarity_search_with_score 查看得分，调试用
        # 方案 C: 使用 max_marginal_relevance_search (MMR) 避免结果过于雷同
        
        try:
            # ✅ 修改：使用重写后的 search_query 进行检索，而不是原始 question
            docs = self.vector_store.max_marginal_relevance_search(
                search_query,  # <--- 这里改为 search_query
                k=k,               
                fetch_k=20,        
                lambda_mult=0.6    
            )
        except Exception as e:
            logger.warning(f"MMR 搜索不可用，降级为普通相似度搜索: {e}")
            # 降级时也用 search_query
            docs = self.vector_store.similarity_search(search_query, k=k)
        
        # 2. 构建结构化上下文
        context_evidence = []
        for i, d in enumerate(docs):
            evidence_block = f"[依据 {i+1}]:\n{d.page_content}"
            context_evidence.append(evidence_block)
        
        context_text = "\n\n".join(context_evidence)
        
        # 3. 构建“强势文化”规则型 Prompt
        system_instruction = """
        # Role: 天道规律解析者 (Expert of The Way)

        ## 核心公理 (Axioms)
        1. **强势文化定义**：强势文化是遵循事物规律的文化，弱势文化是依赖强者的道德期望破格获取的文化。
        2. **实事求是**：一切回答必须基于客观事实（提供的参考上下文），严禁脱离文本的主观臆测或道德说教。
        3. **因果律**：任何现象背后必有其成因，任何结果必有其条件。回答需揭示“条件->结果”的逻辑链条。

        ## 任务协议 (Protocol)
        当用户提出问题时，你必须严格执行以下步骤：

        ### 第一步：事实锚定 (Fact Anchoring)
        - 审查【参考上下文】，提取与问题直接相关的原文片段。
        - **约束**：如果上下文中没有直接答案，明确指出“书中未直接记载”，但可基于书中已确立的“文化属性”逻辑进行推导。严禁编造情节。

        ### 第二步：逻辑推演 (Logical Deduction)
        - 运用“强势文化”视角分析提取的事实。
        - 剖析人物行为背后的文化属性（是强势还是弱势？）。
        - 揭示事件发展的必然性（为什么在这个条件下，必然产生这个结果？）。

        ### 第三步：结构化输出 (Structured Output)
        请严格按照以下格式输出，不要有任何多余的寒暄：

        ---
        ### 📖 原文依据
        > (在此处引用 1-3 句最核心的原文，注明大致章节或情境，作为回答的基石)

        ### 💡 规律解读
        (在此处进行深入分析。不要只复述情节，要回答“为什么”。指出其中的因果逻辑、文化属性冲突以及必然结局。语言风格要冷峻、理性、深刻，类似丁元英的思维模式。)

        ### 🔚 结论
        (用一句话总结该问题反映的天道规律。)
        ---

        ## 用户输入
        【参考上下文】:
        {context}

        【用户问题】:
        {question}

        ## 开始执行推演：
        """

        prompt_text = system_instruction.format(context=context_text, question=question)
        
        # 4. 调用 LLM 流式接口
        logger.info("[规则引擎] 正在基于事实进行逻辑推演...")
        try:
            for chunk in self.llm.stream(prompt_text):
                yield chunk
                
        except Exception as e:
            logger.error(f"LLM 推演失败: {e}")
            yield f"\n\n❌ **逻辑推演中断**: {str(e)}"    


    # get_stats 保持不变
    def get_stats(self) -> Dict[str, Any]:
        count = 0
        if self.vector_store:
            try:
                count = self.vector_store._collection.count()
            except:
                pass
        return {
            "text_loaded": len(self.full_text) > 0,
            "vector_count": count,
            "model": self.model_name,
            "embed_model": self.embed_model_name
        }