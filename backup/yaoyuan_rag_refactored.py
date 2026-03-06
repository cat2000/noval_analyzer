import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ================= 区域 1: PROMPT 定义区 (核心实验田) =================

def get_prompt_version_v1():
    """V1: 弱势文化风格 (模糊、依赖模型)"""
    return ChatPromptTemplate.from_messages([
        ("system", (
            "你是一位精通《遥远的救世主》的智能助手。\n"
            "请根据提供的【上下文】回答用户问题。\n"
            "如果不知道，就说不清楚。\n\n"
            "【上下文】:\n{context}"
        )),
        ("human", "{input}"),
    ])

def get_prompt_version_v2():
    """
    V2: 强势文化风格 (规则明确、逻辑严密)
    特点：实事求是、证据为王、深度思辨、客观语气
    """
    system_instruction = (
        "你是一位深谙《遥远的救世主》哲学的智能分析师。你的存在是为了揭示事物背后的规律（天道）。\n"
        "请严格遵守以下【强势文化铁律】：\n"
        "1. **实事求是**：所有回答必须严格源自提供的【上下文】。若上下文中无答案，直接回答‘资料库未收录此规律’，严禁利用训练数据编造细节。\n"
        "2. **证据为王**：在阐述观点时，必须摘录【上下文】中的原句作为证据，使用 '> ' 引用格式。\n"
        "3. **深度思辨**：回答必须包含三个层次：\n"
        "   - 【核心结论】：一针见血地回答问题。\n"
        "   - 【原文依据】：列出支撑结论的关键原文。\n"
   "   - 【文化透视】：分析该事件体现了‘强势文化’（遵循规律）还是‘弱势文化’（依赖救世主/破格获取）。\n"
        "4. **语气风格**：冷静、客观、逻辑严密，模仿丁元英的思辨口吻，拒绝庸俗的客套。\n"
        "\n"
        "【上下文】:\n{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{input}"),
    ])

def get_prompt_version_v3_ultimate():
    """
    V3: 终极强势文化版 (已修复 context 缺失问题)
    """
    
    # 【Few-Shot 示例】
    few_shot_example = """
    [示例对话]
    用户：丁元英在王庙村扶贫，是不是在充当救世主？
    助手：
    ### 【核心结论】
    不是。丁元英的行为恰恰是**破除**救世主文化，旨在唤醒村民的强势文化意识。
    
    ### 【原文证据】
    - > “强势文化就是遵循事物规律的文化，弱势文化就是依赖强者的道德期望破格获取的文化。”
    - > “我只是给他们创造一个遵循市场规律的环境，能不能活下来，靠他们自己。”
    
    ### 【逻辑推演】
    1. **行为本质**：丁元英没有直接给予金钱施舍，而是构建了市场竞争环境。
    2. **因果规律**：他利用的是市场规律的“道”，而非个人道德的“神”。
    3. **最终目的**：迫使村民在生存压力下独立自强。
    
    ### 【文化透视】
    此举体现了极致的**强势文化**：神即道，道法自然，如来。
    """

    # 【关键修复】在这里加入了 {context} 占位符
    system_instruction = f"""
    # Role (角色定位)
    你是一位深谙《遥远的救世主》哲学的**首席文化分析师**。你的思维内核是**丁元英**：冷峻、客观、洞察人性、遵循天道。

    # Context Data (检索到的原文依据)
    以下是从《遥远的救世主》全书中检索到的相关片段，请严格基于此进行分析：
    <context>
    {{context}}
    </context>

    # Constraints (铁律)
    1. **实事求是**：所有事实性陈述**必须**源自上述 <context> 标签内的内容。
    2. **严禁幻觉**：若 <context> 中无相关信息，必须回答：“资料库未收录此规律，无法基于原文分析。” **严禁**编造书中未提及的情节。
    3. **语气风格**：冷静、理性、简练。禁止使用“亲爱的用户”、“希望帮到您”、“也许”等词汇。
    4. **词汇禁忌**：禁止使用“帮助”、“可怜”等弱势文化词汇；必须使用“规律”、“因果”、“条件”等客观词汇。

    # Instruction (思维链)
    请按以下步骤思考并输出：
    1. **验证**：检查 <context> 中是否有答案。无则直接报错。
    2. **提取**：摘录支撑观点的原句。
    3. **推演**：分析背后的文化属性（强势/弱势）和因果逻辑。
    4. **输出**：严格按下方格式生成。

    # Output Format (输出格式)
    ### 【核心结论】
    (一针见血地回答问题)

    ### 【原文证据】
    - > 引用原文片段 1
    - > 引用原文片段 2

    ### 【逻辑推演】
    1. (行为本质分析)
    2. (因果规律分析)
    3. (深层动因分析)

    ### 【文化透视】
    (总结文化属性，升华到“天道”高度)

    # Few-Shot Example (完美示例)
    {few_shot_example}

    # User Input (当前问题)
    {{input}}
    """
    # 注意：
    # 1. {{context}} 和 {{input}} 使用了双大括号，因为在 Python f-string 中需要转义，
    #    这样 LangChain 才能识别为 {context} 和 {input} 变量。
    # 2. 确保 <context> 标签包裹了 {{context}}，方便模型区分指令和数据。

    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{input}"),
    ])

# ================= 区域 2: CORE LOGIC (纯逻辑执行区) =================

def initialize_components(config):
    """初始化所有组件 (DB, Embedding, LLM)，并确保数据入库"""
    print(f"🚀 正在初始化系统组件...")
    
    documents = []
    # 1. 加载文档
    if os.path.exists(config['text_file']):
        print(f"📖 正在加载文件: {config['text_file']} ...")
        try:
            loader = TextLoader(config['text_file'], encoding='utf-16')
            documents = loader.load()
            print(f"✅ 成功加载 {len(documents)} 个文档块")
        except Exception as e:
            print(f"⚠️ UTF-16 加载失败，尝试 UTF-8... ({e})")
            try:
                loader = TextLoader(config['text_file'], encoding='utf-8')
                documents = loader.load()
                print(f"✅ 成功加载 (UTF-8) {len(documents)} 个文档块")
            except Exception as e2:
                print(f"❌ 文件加载彻底失败: {e2}")
                raise e2
    else:
        raise FileNotFoundError(f"❌ 找不到文件: {config['text_file']}，请确认文件路径正确。")

    # 2. 文本切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""]
    )
    splits = splitter.split_documents(documents)
    print(f"🔪 分割完成，共 {len(splits)} 个片段")

    # 3. Embedding & VectorDB
    print(f"🧠 正在加载嵌入模型: {config['embedding_model']} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config['embedding_model'],
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"💾 正在连接向量数据库: {config['chroma_dir']} ...")
    vectordb = Chroma(
        persist_directory=config['chroma_dir'],
        embedding_function=embeddings,
        collection_name="yaoyuan_collection"
    )

    # 4. 存入数据
    current_count = vectordb._collection.count()
    if current_count == 0:
        print("📝 数据库为空，正在存入全文向量数据 (首次运行可能需要几分钟)...")
        vectordb.add_documents(splits)
        print(f"✅ 数据存入完成，当前库容量: {vectordb._collection.count()}")
    else:
        print(f"✅ 数据库已有 {current_count} 条数据，跳过存入步骤。")

    # 5. LLM
    print(f"🤖 正在连接大模型: {config['llm_model']} ...")
    llm = OllamaLLM(model=config['llm_model'], temperature=0.3)
    
    return vectordb, llm, embeddings

def run_rag_pipeline(vectordb, llm, query, prompt_template):
    """
    执行真实的 RAG 流程 (检索 + 生成)
    """
    # 设置检索器 (k=5 获取更多上下文以支持深度分析)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    # 创建文档组合链
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    
    # 创建检索链
    retrieval_chain = create_retrieval_chain(retriever, combine_chain)
    
    # 执行
    result = retrieval_chain.invoke({"input": query})
    return result["answer"]

# ================= 区域 3: MAIN EXECUTION (交互式循环区) =================

if __name__ == "__main__":
    # 配置
    CONFIG = {
        "text_file": "./遥远的救世主.txt",
        "chroma_dir": "./chroma_db_data",
        "llm_model": "qwen3",
        "embedding_model": "BAAI/bge-m3",
        "chunk_size": 500,
        "chunk_overlap": 50
    }

    # 1. 初始化系统 (只运行一次)
    try:
        vectordb, llm, embeddings = initialize_components(CONFIG)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"💥 初始化失败: {e}")
        sys.exit(1)

    # 2. 选择 Prompt 版本 (在此处切换实验策略)
    current_prompt = get_prompt_version_v3_ultimate() 

    # 3. 启动交互式循环
    print("\n" + "="*60)
    print("🎉 《遥远的救世主》强势文化分析系统已就绪")
    print("💡 提示：输入 'quit' 或 'exit' 退出程序")
    print("="*60 + "\n")

    while True:
        try:
            # 获取用户输入
            user_input = input("📖 请输入问题: ").strip()

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("👋 再见！愿天道酬勤。")
                break
            
            # 跳过空输入
            if not user_input:
                continue

            print("\n🤔 正在检索全书并思考...\n")

            # 执行 RAG
            response = run_rag_pipeline(
                vectordb=vectordb, 
                llm=llm, 
                query=user_input, 
                prompt_template=current_prompt
            )
            
            # 输出结果
            print("-" * 40)
            print(f"💡 分析结果:\n{response}")
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断程序。")
            print("👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("💡 请检查日志或尝试重新提问。\n")
            # 生产环境中建议记录详细 traceback，这里为了简洁只显示错误信息
            # import traceback
            # traceback.print_exc()