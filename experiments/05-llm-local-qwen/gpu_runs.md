# Colab T4 runbook: LLM QA

Cells to paste in order (runtime: GPU T4, 15 GB VRAM).

```bash
# 1. setup (upload or clone the project so that data/processed/*.jsonl exist)
pip install -q "transformers>=5" accelerate bitsandbytes peft scikit-learn pyyaml
cd /content/proyek_akhir
```

## Single run (7B, 4-bit)
```bash
python experiments/global/run.py mrc.llm_qa --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --load-4bit \
    --shots 3 --demo-mode retrieved --prompt-id id1 --batch-size 8 --run-name qwen7b_id1_3shot_retr
```
Full dev (803) at ~0.3 to 0.6 s/example for 0-shot, more for k-shot (longer prompts). Run on `dev` only.

## Prompt sensitivity: 5 prompts x 3 demo orders (fixed 3-shot)
```bash
M=Qwen/Qwen2.5-7B-Instruct
for P in id1 id2 id3 id4 id5; do
  for O in 0 1 2; do
    python experiments/global/run.py mrc.llm_qa --model-name $M --device cuda --load-4bit --shots 3 \
      --prompt-id $P --demo-order-seed $O --batch-size 8
  done
done
# also the English variant and the 0-shot row
python experiments/global/run.py mrc.llm_qa --model-name $M --device cuda --load-4bit --shots 0 --prompt-id en1
```
Each run writes `experiments/05-llm-local-qwen/runs/<auto name>/{config,metrics_dev}.json, preds_dev.jsonl`.
Report mean, std and range of F1 over the 15 cells; that is the prompt-sensitivity number.

## Model comparison (all verified to exist on the Hub on 2026-11-13)
```bash
for M in Qwen/Qwen2.5-7B-Instruct sail/Sailor2-8B-Chat SeaLLMs/SeaLLMs-v3-7B-Chat \
         GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct aisingapore/gemma2-9b-cpt-sea-lionv3-instruct; do
  python experiments/global/run.py mrc.llm_qa --model-name $M --device cuda --load-4bit --shots 3 --batch-size 8
done
```
Gated (need HF token + license): `google/gemma-2-9b-it`, `meta-llama/Llama-3.1-8B-Instruct`.
If a model has no chat template or ignores the instruction, check `raw_output` in preds first.

## Contamination probe
```bash
python experiments/05-llm-local-qwen/llm_contamination.py --n 50 --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --load-4bit
```

## QLoRA fine-tune then evaluate
```bash
python experiments/05-llm-local-qwen/llm_lora.py --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --load-4bit \
   --epochs 1 --batch-size 4 --grad-accum 4 --lr 2e-4 --out-dir experiments/05-llm-local-qwen/lora_qwen7b
python experiments/global/run.py mrc.llm_qa --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --load-4bit \
   --adapter-dir experiments/05-llm-local-qwen/lora_qwen7b/adapter --shots 0 --prompt-id id1 --run-name qwen7b_lora_0shot
```
Expect about 1.5 to 3 h per epoch on a T4 for 6435 examples at max-len 768; use `--max-steps 600` for a quick pass.

## Threshold tuning
`--na-method seqlogprob` is the default (one extra forward per example; add `--na-method first_token` for the old first-token probability). Check the scorer first with `python experiments/global/run.py mrc.llm_qa --self-test --model-name <M> --device cuda`.
`na_prob` is in every preds file. `python experiments/global/run.py mrc.evaluate --gold data/processed/dev.jsonl --pred <preds> --na-threshold-search`.
Threshold is tuned on dev; do not search on itest.
