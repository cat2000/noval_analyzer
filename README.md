📚 《遥远的救世主》读书小精灵 (RAG Knowledge Assistant)
一个基于 Local LLM 和 RAG (检索增强生成) 技术的智能读书助手，专为深度研读豆豆著作《遥远的救世主》而设计。
本项目不仅仅是一个问答机器人，更是一个可观测、可调节、具备自我反思能力的检索实验平台。它融合了丁元英式的“直击本质”思维与严谨的学术引用规范，帮助用户在海量文本中快速定位核心逻辑与文化属性。
神即道，道法自然，如来。
让 AI 带你参悟强势文化的逻辑闭环。
✨ 核心特性
🧠 智能检索引擎 (RAG Engine)
混合检索策略 (Hybrid Retrieval)：结合 BM25 (关键词匹配) 与 Vector Search (语义向量)，通过加权融合算法，确保既不漏掉专有名词，也能理解深层语义。
重排序优化 (Rerank)：引入 Cross-Encoder 模型 (BAAI/bge-reranker-large) 对初筛结果进行精细打分，剔除噪声，只保留最相关的上下文。
Self-RAG 机制：内置质量评估模块。若检索内容相关性低，系统会自动触发重试机制或降级策略（如调整查询改写模式），确保回答有据可依。
流式生成 (Streaming)：基于 Ollama 本地大模型，支持打字机效果的实时输出，首字延迟极低。
智能去重与引用：自动识别并去除回答中重复问题的首行，强制要求所有观点必须标注原文出处（章节与依据编号），杜绝幻觉。
🎨 交互式 Web 界面 (Streamlit UI)
全链路可视化监控：独创的Pipeline 节点监控条，实时展示 语义重写 -> 多路发散 -> 混合检索 -> 重排序 -> 质量评估 -> 文章生成 每个阶段的耗时。
动态瓶颈高亮：自动识别耗时最长的环节（瓶颈节点），并以红色高亮显示，提供直观的调优建议。
实时参数调优：用户可在界面上直接调整 TopK、重排阈值、改写策略 等核心参数，即时观察对检索效果的影响，无需重启服务。
证据溯源面板：右侧边栏实时展示被引用的原文片段，高亮显示“核心依据”与“辅助参考”，让每一个结论都经得起推敲。
禅意交互体验：加载与思考过程中融入《遥远的救世主》经典语录与禅意思考动画，沉浸感拉满。
🛠️ 技术栈
表格
组件	技术选型	说明
大语言模型	Qwen3 (via Ollama)	本地部署，负责推理、改写与生成，保障数据隐私
嵌入模型	BAAI/bge-large-zh	高精度中文向量模型，构建语义空间
重排序模型	BAAI/bge-reranker-large	Cross-Encoder 架构，精准评估 Query-Doc 相关性
向量数据库	ChromaDB	轻量级、持久化向量存储，支持增量索引
应用框架	Streamlit + LangChain	快速构建数据驱动的 Web 交互界面
检索算法	BM25 + Cosine Similarity	混合检索，兼顾精确匹配与模糊语义
知识库源	《遥远的救世主》.txt	豆豆著，关于文化属性与强势文化的经典著作
🚀 快速开始
1. 环境准备
确保您的系统已安装 Python 3.9+ 和 Ollama。
安装 Ollama 并拉取模型
bash

编辑



# 访问 https://ollama.com 安装 Ollama
# 拉取所需模型
ollama pull qwen3
ollama pull qwen3:0.6b  # 用于快速评估的小模型
安装 Python 依赖
bash

编辑



pip install -r requirements.txt
(注：请确保 requirements.txt 包含 langchain, chromadb, sentence-transformers, streamlit, scikit-learn, numpy 等库)
2. 准备数据
将小说文本文件命名为 遥远的救世主.txt 并放置在项目根目录下。
3. 启动应用
bash

编辑



streamlit run app.py
首次启动时，系统会自动构建向量索引，可能需要几分钟时间，请耐心等待进度条完成。
⚙️ 核心参数详解
在 Web 界面的“链路监控与调优”面板中，您可以实时调整以下参数：
表格
参数名	含义	推荐值	影响
改写策略 (rewrite_mode)	Query 重写模式	deep	direct: 原样查询; light: 轻量清洗; deep: 深度语义扩充
发散路数 (multi_query_count)	多路查询变体数量	1	生成多个角度的查询语句并行检索，提高召回率，但增加耗时
初召 TopK (top_k_initial)	混合召回数量	8	从向量库和 BM25 中初步召回的文档总数
重排门槛 (rerank_threshold)	相关性过滤阈值	0.5	低于此分数的片段将被直接丢弃，宁缺毋滥
最大重试 (max_self_rag_attempts)	自动重试次数	0	检索质量不佳时自动重新检索的次数。设为 0 可禁用以提升速度
终选 TopK (top_k_final)	最终送入上下文片段数	3	经过 Rerank 后，最终交给 LLM 阅读的文档数量
📂 项目结构
text

编辑



noval_analyzer/
├── app.py                  # Streamlit 前端入口 (文件二)
├── rag_engine.py           # 核心 RAG 引擎逻辑 (文件一)
├── 遥远的救世主.txt         # 知识库源文件
├── chroma_db/              # 向量数据库持久化目录 (自动生成)
├── logs/                   # 会话日志与性能指标 (自动生成)
├── checkpoint.json         # 索引构建断点记录 (自动生成)
├── requirements.txt        # Python 依赖列表
└── README.md               # 项目说明文档
📊 可观测性与日志
系统内置了完善的日志记录机制：
控制台日志：实时输出各阶段耗时、检索得分、重试触发情况。
JSONL 会话日志：保存在 logs/rag_session_YYYYMMDD.jsonl。
记录每次问答的完整上下文、引用情况、性能指标。
自动标记潜在风险：hallucination_risk (无引用)、low_utilization (低利用率)、high_latency (高延迟)。
💡 设计理念
拒绝废话：Prompt 工程强制模型开篇即结论，严禁“根据上下文”、“综上所述”等冗余表述。
文雅排版：关键概念加粗，原文引用使用块引用格式，保持阅读美感。
分数优先：在评估环节，优先信任 Rerank 模型的数值评分，仅在模糊地带调用小模型辅助，实现速度与精度的最佳平衡。
透明可控：将黑盒的 RAG 过程白盒化，让用户清楚看到每一步发生了什么，并能亲手干预。
🤝 贡献与反馈
本项目旨在探索本地化 RAG 在垂直文学领域的最佳实践。如果您有任何优化建议或发现了 Bug，欢迎提交 Issue 或 Pull Request。
特别致谢：
豆豆 著《遥远的救世主》
LangChain Community
Hugging Face BGE Models
Ollama Team
Built with ❤️ and Logic.