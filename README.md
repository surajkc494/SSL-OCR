# SSL-OCR (Telugu Handwritten Word Recognition)

This repository is adapted for **Telugu handwritten word recognition** with a CTC objective and grapheme-level vocabulary.

## Dataset Layout

```text
images/                   # all .jpg handwritten word images
labels.csv                # columns: image_id,text
vocab.txt                 # one line: <grapheme>\t<index>
train.txt                 # one line: <image_id> <idx1> <idx2> ...
valid.txt
test.txt
```

### Notes
- `image_id` in `labels.csv` is authoritative for image lookup (`image_id + '.jpg'`).
- `text` is already clean Telugu word label.
- UTF-8 is required for all Telugu text files.

## Install

```bash
pip install torch torchvision pillow pandas numpy tqdm editdistance
```

## Train

```bash
export PYTHONIOENCODING=utf-8
python train.py \
  --vocab_file vocab.txt \
  --train_file train.txt \
  --val_file valid.txt \
  --labels_csv labels.csv \
  --image_dir ./images/ \
  --img_height 64 \
  --max_width 512 \
  --batch_size 32 \
  --num_workers 4
```

## Fine-tune

```bash
python fine_tune.py \
  --vocab_file vocab.txt \
  --train_file train.txt \
  --val_file valid.txt \
  --labels_csv labels.csv \
  --image_dir ./images/ \
  --pretrained_encoder_path /path/to/encoder.pt
```

## Test

```bash
export PYTHONIOENCODING=utf-8
python test.py \
  --vocab_file vocab.txt \
  --test_file test.txt \
  --labels_csv labels.csv \
  --image_dir ./images/ \
  --test_model ./weights/best_telugu_ctc.pt
```

Predictions are written to `pred_logs/test_predictions.tsv`.

## Key Telugu Handling Rules
- Tokenization is grapheme-level via `TeluguVocab.encode()` (greedy longest match).
- CTC blank index is `vocab_size`.
- CER is computed on grapheme token sequences, not Unicode codepoints.
