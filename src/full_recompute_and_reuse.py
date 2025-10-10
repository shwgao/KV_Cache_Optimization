#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import yaml  # type: ignore
import torch  # type: ignore
from transformers import (AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer,
                          BitsAndBytesConfig)  # type: ignore
# 指标计算依赖
# from datasets import load_metric  # ROUGE指标
from sklearn.metrics import f1_score  # F1指标
from sklearn.preprocessing import LabelEncoder  # F1标签编码

# Local modules
from rag_retrieval import RetrievalConfig, ColbertRetrieval  # type: ignore
from build_kv_cache import extract_texts  # type: ignore
from evaluate import load

# --------------------------- 1. 指标计算工具函数（核心） ---------------------------
def init_rouge_metric():
    """初始化ROUGE指标（适配摘要任务，用于MultiNews）"""
    try:
        return load("rouge")  # 使用evaluate库的load函数
    except ImportError:
        raise ImportError("请安装evaluate库：pip install evaluate")


def calculate_rouge(rouge_metric, pred, ref):
    results = rouge_metric.compute(predictions=[pred], references=[ref])
    return {
        "rouge-1-f1": round(results["rouge1"], 4),
        "rouge-2-f1": round(results["rouge2"], 4),
        "rouge-l-f1": round(results["rougeL"], 4)
    }


def init_f1_encoder():
    """初始化F1指标的标签编码器（适配分类任务，用于非MultiNews数据集）"""
    return LabelEncoder()


def calculate_f1(pred: str, ref: str, encoder: LabelEncoder) -> tuple[dict, LabelEncoder]:
    pred = pred.strip() if pred and pred.strip() else "unknown_pred"
    ref = ref.strip() if ref and ref.strip() else "unknown_ref"

    # 如果encoder尚未fit，则初始化classes_
    if not hasattr(encoder, "classes_"):
        encoder.fit([pred, ref])
    else:
        # 避免fit时丢掉旧标签
        all_labels = list(encoder.classes_) + [pred, ref]
        encoder.fit(list(set(all_labels)))  # 去重

    pred_enc = encoder.transform([pred])
    ref_enc = encoder.transform([ref])
    f1 = f1_score(ref_enc, pred_enc, average="macro", zero_division=0)
    return {"f1-score": round(f1, 4)}, encoder

def is_multinews(input_path: str) -> bool:
    """自动识别是否为MultiNews数据集（基于文件名或样本字段）"""
    # 1. 文件名判断（含"multinews"关键词）
    if "multinews" in input_path.lower():
        return True
    
    # 2. 样本字段判断（MultiNews含"reference_summary"和"texts/document"字段）
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            first_sample = data[0] if isinstance(data, list) else data
            if "reference_summary" in first_sample and (("texts" in first_sample) or ("document" in first_sample)):
                return True
    except Exception:
        pass
    
    return False


# --------------------------- 2. I/O工具函数 ---------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 适配多种多种输入格式
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "samples" in data:
            return data["samples"]
        if "results" in data:
            return data["results"]
    return [data]


# --------------------------- 3. 检索函数 ---------------------------
def run_retrieval(samples: List[Dict[str, Any]], cfg: Dict[str, Any], top_k: int) -> None:
    """运行ColBERT检索，为样本添加retrieved_indices和retrieved_scores字段"""
    retrieval_cfg = RetrievalConfig(** cfg.get("retrieval", {}))
    if not getattr(retrieval_cfg, "checkpoint", None):
        retrieval_cfg.checkpoint = getattr(retrieval_cfg, "model_id", "colbert-ir/colbertv2.0")
    
    retrieval = ColbertRetrieval(retrieval_cfg)
    retrieval.prepare(samples)
    retrieval.retrieve(samples, top_k=top_k)


# --------------------------- 4. Prompt构建（适配MultiNews） ---------------------------
def build_prompt_from_topk(sample: Dict[str, Any], top_k: int) -> Tuple[str, str]:
    """构建上下文和指令，适配MultiNews的文档结构"""
    MAX_CTX_CHARS = 2000
    # 适配MultiNews的文档字段
    if "document" in sample:
        text_pairs = [(i, doc) for i, doc in enumerate(sample["document"])]
    elif "texts" in sample:
        text_pairs = [(i, txt) for i, txt in enumerate(sample["texts"])]
    else:
        text_pairs = extract_texts(sample)
    
    idx2text = {i: t for i, t in text_pairs}
    retrieved_indices = [int(i) for i in sample.get("retrieved_indices", [])]
    selected_indices = retrieved_indices[:top_k] if retrieved_indices else list(idx2text.keys())[:top_k]
    
    # 拼接上下文（去重+过滤空文本）
    context_chunks = [idx2text[i] for i in selected_indices if i in idx2text and idx2text[i].strip()]
    context = "\n\n".join(context_chunks) if context_chunks else "No context available."
    if len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS]
    
    # 适配指令：摘要任务 vs QA任务
    default_qa_instruct = "Use the following context to answer the question clearly."
    question = sample.get("question", default_qa_instruct).strip()

    print("sample_id", sample.get("id"), "retrieved_indices", sample.get("retrieved_indices"))

    print("len of ctx:", len(context))
    return context, question

def encode_input(tokenizer, context: str, question: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:  # 修改返回类型
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msg_content = f"{question}\n\nContext: {context}" if "summarize" in question.lower() else f"Context: {context}\n\nQuestion: {question}"
        messages = [{"role": "user", "content": msg_content}]
        # 生成attention_mask
        encoding = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True  # 启用attention_mask
        )
            # 兼容可能返回 tensor 或 dict
        if isinstance(encoding, dict):
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
        elif torch.is_tensor(encoding):
            input_ids = encoding
            attention_mask = torch.ones_like(input_ids)
        else:
            raise ValueError(f"Unknown encoding type: {type(encoding)}")

    else:
        prompt = f"{question}\n\nContext: {context}\n\nSummary:" if "summarize" in question.lower() else f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        encoding = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=tokenizer.model_max_length // 2,
            return_attention_mask=True  # 启用attention_mask
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
    
    # 转移设备（两者都要移到目标设备）
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    return input_ids, attention_mask  # 返回两个张量

# --------------------------- 5. 模型生成函数 ---------------------------
def decode_full_recompute(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor, 
    max_new_tokens: int,
) -> Dict[str, Any]:
    """全量预计算生成，返回生成结果和时序指标"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 输入设备对齐
    # if getattr(model, "hf_device_map", None) is None:
    #     input_ids = input_ids.to(next(model.parameters()).device, non_blocking=True)
    
    # 生成配置
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    
    # 用Streamer跟踪首token时间
    print("generate streamer")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer
    
    # 线程化生成
    import threading
    first_token_time: Optional[float] = None
    start_time = time.perf_counter()

    def _generate():
        with torch.inference_mode():
            print("start to generate data")
            model.generate(** generation_kwargs)

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    # 收集生成结果
    generated_chunks: List[str] = []
    try:
        for chunk in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated_chunks.append(chunk)
    finally:
        thread.join()

    # 计算时序指标
    end_time = time.perf_counter()
    generated_text = "".join(generated_chunks).strip()
    try:
        generated_token_num = len(tokenizer.encode(generated_text, add_special_tokens=False))
    except Exception:
        generated_token_num = max(1, len(generated_chunks))

    ttft = (first_token_time - start_time) if first_token_time else 0.0
    e2e_latency = end_time - start_time
    throughput = (generated_token_num / e2e_latency) if e2e_latency > 0 else 0.0
    tpot = (e2e_latency / generated_token_num) if generated_token_num > 0 else 0.0

    return {
        "answer": generated_text,
        "generated_token_num": generated_token_num,
        "ttft": round(ttft, 4),
        "e2e_latency": round(e2e_latency, 4),
        "throughput": round(throughput, 4),
        "tpot": round(tpot, 4)
    }


# --------------------------- 6. 主函数（完整逻辑） ---------------------------
def main() -> None:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成+时序指标+ROUGE/F1指标，结果写入文件")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--input", type=str, default="inputs/musique_s.json", help="输入数据集路径")
    parser.add_argument("--output", type=str, default="results/full_kv_recompute_results", help="输出目录")
    parser.add_argument("--top_k", type=int, default=10, help="检索/使用的文本片段数量")
    parser.add_argument("--retrieval_json", type=str, default="retrieval_topk.json", help="检索结果文件名")
    parser.add_argument("--use_quantization", action="store_true", help="启用量化（4/8位）")
    parser.add_argument("--quant_bit", type=int, choices=[4, 8], default=8, help="量化位宽")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="最大生成token数")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称")
    args = parser.parse_args()

    # 1. 初始化输出目录+删除旧检索文件
    os.makedirs(args.output, exist_ok=True)
    retrieval_path = os.path.join(args.output, args.retrieval_json)
    if os.path.exists(retrieval_path):
        os.remove(retrieval_path)
        print(f"已删除旧检索文件：{retrieval_path}")

    # 2. 加载数据+识别数据集类型
    cfg = load_config(args.config)
    samples = load_samples(args.input)
    dataset_is_multinews = is_multinews(args.input)
    print(f"数据集类型：{'MultiNews（计算ROUGE）' if dataset_is_multinews else '其他（计算F1）'}")
    print(f"待处理样本数：{len(samples)}")

    # 3. 解析核心参数
    model_name = args.model_name
    device = torch.device(cfg.get("model", {}).get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    top_k = cfg.get("retrieval", {}).get("top_k", args.top_k)
    default_gen_tokens = 300 if dataset_is_multinews else 32
    max_new_tokens = args.max_new_tokens or cfg.get("generation", {}).get("max_new_tokens", default_gen_tokens)
    print(f"生成配置：max_new_tokens={max_new_tokens}, device={device}")

    # 4. 加载Tokenizer和Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置
    quantization_config = None
    if args.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(args.quant_bit == 4),
            load_in_8bit=(args.quant_bit == 8),
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        print(f"启用{args.quant_bit}位量化")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        print(f"模型加载完成：{model_name}（设备：{device}")

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None,
            trust_remote_code=True
        ).to(device).eval()
        print(f"模型加载完成：{model_name}（设备：{device}")

    # 5. 运行检索+保存检索结果
    run_retrieval(samples, cfg, top_k)
    retrieval_results = [
        {
            "id": smp.get("id"),
            "retrieved_indices": smp.get("retrieved_indices", []),
            "retrieved_scores": smp.get("retrieved_scores", [])
        }
        for smp in samples
    ]
    with open(retrieval_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
    print(f"检索结果已保存至：{retrieval_path}")

    # 6. 初始化指标工具
    if dataset_is_multinews:
        rouge_metric = init_rouge_metric()
        print("已初始化ROUGE指标计算器")
    else:
        f1_encoder = init_f1_encoder()
        print("已初始化F1指标编码器")

    # 7. 样本推理+指标计算
    results: List[Dict[str, Any]] = []
    qa_results: List[Dict[str, str]] = [] 
    results_reuse: List[Dict[str, Any]] = []
    qa_results_reuse: List[Dict[str, str]] = [] 
    compare_results: List[Dict[str, Any]] = []
    
    for idx, sample in enumerate(samples):
        sid = sample.get("id", str(idx))
        # try:
        # 构建输入
        context, question = build_prompt_from_topk(sample, top_k)
        print("start to encode ")
        input_ids, attention_mask  = encode_input(tokenizer, context, question, device)
        # print("tokens:", inputs["input_ids"].shape[1])
        
        # 模型生成
        decode_result = decode_full_recompute(model, tokenizer, input_ids, attention_mask, max_new_tokens)  # 传入attention_mask
        decode_result["sample_id"] = sid  # 添加样本ID
        print("finish compute")
        qa_results.append({
            "sample_id": sid,
            "question": question,
            "reference_answer": sample.get("answer", ""),
            "generated_answer": decode_result["answer"]
        })
        # 计算并添加指标
        if dataset_is_multinews:
            # MultiNews：使用reference_summary计算ROUGE
            ref_text = sample.get("reference_summary", "")
            metrics_recompute  = calculate_rouge(rouge_metric, decode_result["answer"], ref_text)
            decode_result.update(metrics_recompute )
        else:
            # 其他数据集：使用answer作为参考计算F1（可根据实际场景修改参考字段）
            ref_text = sample.get("answer", "")  # 假设参考答案在"answer"字段
            metrics_recompute , f1_encoder = calculate_f1(decode_result["answer"], ref_text, f1_encoder)
            decode_result.update(metrics_recompute )
        
        results.append(decode_result)
        print(f"已处理 {idx+1}/{len(samples)} 样本（ID: {sid}）")

        # ---------------------- full reuse ----------------------
        # 这里模拟重用: 直接用之前的 context 输入，不重新检索或处理
        decode_result_reuse = decode_full_recompute(model, tokenizer, input_ids, attention_mask, max_new_tokens)
        decode_result_reuse["sample_id"] = sid
        qa_results_reuse.append({
            "sample_id": sid,
            "question": question,
            "reference_answer": sample.get("answer", ""),
            "generated_answer": decode_result_reuse["answer"]
        })
        if dataset_is_multinews:
            metrics_reuse = calculate_rouge(rouge_metric, decode_result_reuse["answer"], ref_text)
            decode_result_reuse.update(metrics_reuse)
        else:
            metrics_reuse, f1_encoder = calculate_f1(decode_result_reuse["answer"], ref_text, f1_encoder)
            decode_result_reuse.update(metrics_reuse)

        results_reuse.append(decode_result_reuse)

        # ---------------------- 对比 ----------------------
        compare_results.append({
            "sample_id": sid,
            "question": question,
            "reference_answer": sample.get("answer", ""),
            "full_recompute_answer": decode_result["answer"],
            "full_recompute_metrics": {k: decode_result[k] for k in metrics_recompute.keys()},
            "full_reuse_answer": decode_result_reuse["answer"],
            "full_reuse_metrics": {k: decode_result_reuse[k] for k in metrics_reuse.keys()},
        })
        
        # except Exception as e:
        #     error_msg = f"处理样本 {sid} 出错: {str(e)[:100]}"
        #     print(error_msg)
        #     # 错误样本也记录结果，方便后续排查
        #     error_result = {
        #         "sample_id": sid,
        #         "answer": error_msg,
        #         "generated_token_num": 0,
        #         "ttft": 0.0,
        #         "e2e_latency": 0.0,
        #         "throughput": 0.0,
        #         "tpot": 0.0
        #     }
        #     # 添加空指标，保持结构一致
        #     if dataset_is_multinews:
        #         error_result.update({
        #             "rouge-1-f1": 0.0,
        #             "rouge-2-f1": 0.0,
        #             "rouge-l-f1": 0.0
        #         })
        #     else:
        #         error_result.update({"f1-score": 0.0})
        #     results.append(error_result)

    # 8. 计算并添加平均指标
    if results:
        n = len(results)
        avg_metrics: Dict[str, float] = {
            "sample_id": "average",
            "generated_token_num": round(sum(r["generated_token_num"] for r in results) / n, 1),
            "ttft": round(sum(r["ttft"] for r in results) / n, 4),
            "e2e_latency": round(sum(r["e2e_latency"] for r in results) / n, 4),
            "throughput": round(sum(r["throughput"] for r in results) / n, 4),
            "tpot": round(sum(r["tpot"] for r in results) / n, 4)
        }
        
        # 添加平均指标
        if dataset_is_multinews:
            avg_metrics.update({
                "rouge-1-f1": round(sum(r.get("rouge-1-f1", 0.0) for r in results) / n, 4),
                "rouge-2-f1": round(sum(r.get("rouge-2-f1", 0.0) for r in results) / n, 4),
                "rouge-l-f1": round(sum(r.get("rouge-l-f1", 0.0) for r in results) / n, 4)
            })
        else:
            avg_metrics.update({
                "f1-score": round(sum(r.get("f1-score", 0.0) for r in results) / n, 4)
            })
        
        results.append(avg_metrics)

    # 9. 保存最终结果（包含指标）
    results_path = os.path.join(args.output, "full_recompute_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    qa_results_path = os.path.join(args.output, "qa_results.json")
    with open(qa_results_path, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output, "full_reuse_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_reuse, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output, "compare_results.json"), "w", encoding="utf-8") as f:
        json.dump(compare_results, f, indent=2, ensure_ascii=False)
    print(f"QA实际结果与参考答案已保存至：{qa_results_path}")
    # 输出统计信息
    processed = len(results) - (1 if results and results[-1]["sample_id"] == "average" else 0)
    print(f"\n处理完成！共处理 {processed} 个样本")
    print(f"结果（含指标）已保存至：{results_path}")


if __name__ == "__main__":
    main()
