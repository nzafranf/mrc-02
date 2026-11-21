# Running the PLM QA model on Colab / Kaggle (T4)

## Setup
Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Accelerator = GPU T4 x1, Internet on.

```bash
# option A: upload proyek_akhir.zip (must contain experiments/global/, this experiment folder and data/processed/; never data/processed/itest*) then
unzip -q proyek_akhir.zip -d /content && cd /content/proyek_akhir
# option B: git clone <your repo url> && cd <repo>   (data/processed must be present)

pip install -q "transformers>=5.4" peft sentencepiece accelerate
python -c "import torch;print(torch.cuda.get_device_name(0))"
```
Kaggle (scripted): see `experiments/04-plm-indobert/kaggle/`. Manual Kaggle: use `/kaggle/working` instead of `/content`. If `import peft` fails with a torchvision error, run `pip uninstall -y torchvision`.

Every command writes `experiments/04-plm-indobert/runs/<run-name>/seed<k>/{config.json,preds_dev.jsonl,metrics_dev.json,log.txt}`.
Add `--save-checkpoint` only if you need `--predict-only` later (stores only trainable weights; ~500 MB for full fine-tuning, a few MB for LoRA/probe).
Common flags: `--device cuda --fp16 --epochs 3 --lr 3e-5 --batch-size 16`. If you run out of memory use `--batch-size 8 --grad-accum 2`.

## Offset check (CPU, about 2 minutes each)
```bash
for m in indobenchmark/indobert-base-p1 FacebookAI/xlm-roberta-base indolem/indobert-base-uncased; do
  python experiments/04-plm-indobert/plm_qa.py --check-offsets --model-name $m; done
```

## Main runs, 3 seeds
```bash
for s in 13 42 87; do
  python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --seed $s --run-name plm_indobert_p1 \
    --model-name indobenchmark/indobert-base-p1
done
for s in 13 42 87; do
  python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --seed $s --run-name plm_xlmr \
    --model-name FacebookAI/xlm-roberta-base
done
# optional third encoder
for s in 13 42 87; do
  python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --seed $s --run-name plm_indolem \
    --model-name indolem/indobert-base-uncased
done
```

## Freeze vs full vs LoRA (seed 13; repeat for 42, 87 if time allows)
```bash
M=indobenchmark/indobert-base-p1
python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --run-name plm_full   --model-name $M            # lr 3e-5
python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --run-name plm_probe  --model-name $M --freeze-encoder --lr 1e-3 --epochs 5
python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --run-name plm_lora   --model-name $M --lora --lr 3e-4 --epochs 5
```
Frozen and LoRA need larger learning rates than full fine-tuning; do not reuse 3e-5.

## Data efficiency (train-frac, random subset of context groups)
```bash
for f in 0.1 0.25 0.5 1.0; do
  for s in 13 42 87; do
    python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --seed $s --train-frac $f --run-name plm_frac$f \
      --model-name indobenchmark/indobert-base-p1
  done
done
```
For small fractions consider `--epochs 5` (add it to every fraction for a fair comparison).

## Predict on another split
```bash
python experiments/04-plm-indobert/plm_qa.py --predict-only --checkpoint experiments/04-plm-indobert/runs/plm_full/seed13/checkpoint.pt \
  --split-file data/processed/dev.jsonl --out /tmp/preds.jsonl --device cuda
```
The `na_prob` threshold tuned on dev is stored in the checkpoint (`na_threshold`) and in `metrics_dev.json` (`best_na_threshold`).

## Expected T4 runtimes (estimates, fp16, batch 16, max_len 384, about 7k training windows per epoch)
| Run | per epoch | total (3 epochs) |
|---|---|---|
| IndoBERT-base full | 6 to 9 min | 20 to 30 min incl. dev eval |
| XLM-R base full | 8 to 11 min | 25 to 35 min |
| LoRA r=8 | 5 to 7 min | 5 epochs: 25 to 35 min |
| Frozen encoder (probe) | 2 to 3 min (backward still passes through no encoder) | 5 epochs: 10 to 15 min |
| train-frac 0.1 / 0.25 / 0.5 | scales linearly | 3 to 15 min |

Totals: 3 seeds x 2 encoders about 2.5 h; the train-frac grid (4 x 3 seeds) about 3 h. Colab free sessions are around 4 to 5 h, so split across sessions or use Kaggle (30 GPU h per week).
