import json
import glob
import os
import sys

# 尝试导入 pandas 进行数据分析，如果没有安装则使用原生 Python 处理
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️ 未检测到 pandas 库，将使用基础模式运行 (功能受限)。")
    print("💡 建议安装以获得更好体验：pip install pandas")

def analyze_latest_log():
    # 1. 寻找最新的日志文件
    logs = glob.glob("./logs/rag_session_*.jsonl")
    
    if not logs:
        print("\n❌ 未找到日志文件。")
        print("👉 请先运行主程序 (python app.py) 并进行几次问答，生成日志后再运行此脚本。")
        return
    
    # 按修改时间排序，取最新的一个
    latest_log = max(logs, key=os.path.getctime)
    print(f"\n📊 正在分析最新日志：{latest_log}")
    print("=" * 50)
    
    # 2. 读取数据
    data = []
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"❌ 读取日志失败：{e}")
        return

    if not data:
        print("⚠️ 日志文件为空。")
        return

    # 3. 计算指标
    total_queries = len(data)
    
    # 提取关键指标列表
    latencies = [d['metrics'].get('latency_seconds', 0) for d in data]
    noise_ratios = [d['metrics'].get('noise_ratio', 0) for d in data]
    citation_rates = [d['metrics'].get('citation_rate', 0) for d in data]
    retry_counts = [d['metrics'].get('self_rag_attempts', 1) for d in data]
    
    # 识别风险案例
    hallucination_cases = [d for d in data if d['metrics'].get('cited_count', 0) == 0 and d['metrics'].get('total_retrieved', 0) > 0]
    high_latency_cases = [d for d in data if d['metrics'].get('latency_seconds', 0) > 15.0]
    low_utilization_cases = [d for d in data if d['metrics'].get('citation_rate', 1) < 0.3 and d['metrics'].get('total_retrieved', 0) > 2]

    # 4. 输出报告
    print(f"\n📈 === RAG 系统健康报告 ===")
    print(f"📅 统计样本数：{total_queries} 次问答")
    print(f"⏱️ 平均响应延迟：{sum(latencies)/len(latencies):.2f} 秒")
    print(f"🎯 平均引用率 (Context Utilization): {sum(citation_rates)/len(citation_rates):.2%}")
    print(f"🗑️ 平均噪声比 (Noise Ratio): {sum(noise_ratios)/len(noise_ratios):.2%} (越低越好)")
    print(f"🔄 平均 Self-RAG 重试次数：{sum(retry_counts)/len(retry_counts):.2f} 次")
    
    # 风险统计
    print(f"\n⚠️ === 风险检测 ===")
    print(f"🤥 潜在幻觉案例 (有检索无引用): {len(hallucination_cases)} 起 ({len(hallucination_cases)/total_queries:.2%})")
    print(f"🐢 高延迟案例 (>15s): {len(high_latency_cases)} 起")
    print(f"😴 低效检索案例 (引用率<30%): {len(low_utilization_cases)} 起")

    # 5. 展示坏案例详情 (Top 3)
    if hallucination_cases:
        print(f"\n🔍 === 幻觉风险案例详情 (前 3 例) ===")
        for i, case in enumerate(hallucination_cases[:3], 1):
            print(f"\n[案例 {i}]")
            print(f"❓ 问题：{case['question']}")
            print(f"🤖 回答预览：{case['response_preview']}")
            print(f"📉 检索详情：检索了 {case['metrics']['total_retrieved']} 个文档，但引用了 0 个。")
            print(f"💡 建议：检查 Prompt 中的引用指令，或检查检索内容是否真的相关。")
            print("-" * 40)

    if high_latency_cases and not hallucination_cases:
        # 如果没有幻觉案例，显示一些高延迟的看看
        print(f"\n🐢 === 高延迟案例详情 (前 2 例) ===")
        for i, case in enumerate(high_latency_cases[:2], 1):
            print(f"\n[案例 {i}] 耗时：{case['metrics']['latency_seconds']:.2f}s")
            print(f"❓ 问题：{case['question']}")
            print("-" * 40)

    print("\n✅ 分析完成。")
    print("💡 优化建议:")
    if sum(noise_ratios)/len(noise_ratios) > 0.6:
        print("   - 噪声比过高：建议减小 top_k_initial 或优化 Rerank 模型。")
    if len(hallucination_cases) / total_queries > 0.1:
        print("   - 幻觉率过高：建议在 System Prompt 中加强‘必须引用’的约束，或检查检索质量。")
    if sum(retry_counts)/len(retry_counts) > 1.5:
        print("   - 重试频繁：说明首次检索 (Multi-Query) 效果不佳，尝试优化 Query Rewrite 的 Prompt。")

if __name__ == "__main__":
    analyze_latest_log()