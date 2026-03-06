import os
import json
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
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
                 embed_model_name: str = "nomic-embed-text"):
        
        self.txt_path = txt_path
        self.db_path = db_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        
        # 初始化组件
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        self.llm = OllamaLLM(model=self.model_name, temperature=0.7)
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
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

    def query(self, question: str, k: int = 3):
        """执行检索增强生成 (RAG) - 改为流式生成器"""
        
        # --- 初始化检查 (保持不变) ---
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
        # ----------------------------

        # 1. 检索相关片段
        logger.info(f"正在检索与 '{question}' 相关的上下文...")
        try:
            docs = self.vector_store.similarity_search(question, k=k)
        except Exception as e:
            yield f"❌ 检索出错: {str(e)}"
            return
        
        if not docs:
            yield "⚠️ 未在知识库中找到相关信息，但我可以尝试根据通用知识回答："
            # 即使没找到，也可以继续让 LLM 尝试，或者直接返回
        
        # 2. 构建上下文
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 3. 构建 Prompt
        prompt_text = f"""
        你是一位精通《遥远的救世主》(天道) 的强势文化专家。
        请根据以下【参考上下文】回答用户的问题。
        如果上下文中没有答案，请结合你对原著的理解进行推导，但要注明是推导内容。
        回答风格要深刻、理性，体现“强势文化”的思维逻辑。

        【参考上下文】:
        {context_text}

        【用户问题】:
        {question}

        【专家回答】:
        """
        
        # 4. 【关键修改】调用 LLM 的流式接口并 yield 结果
        try:
            # OllamaLLM 通常支持 stream=True 或者 invoke 返回迭代器
            # LangChain 的 OllamaLLM 通常需要这样调用以获取流：
            for chunk in self.llm.stream(prompt_text):
                yield chunk
                
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            yield f"\n\n❌ **生成过程中出错**: {str(e)}"

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