# SSL-OCR (Telugu Handwritten Word Recognition)

This repository now contains a **Telugu handwritten word OCR pipeline** built on a ViT encoder + CTC recognizer.

> Note on naming: the repository name remains `SSL-OCR` for continuity, but the primary training scripts (`train.py`, `fine_tune.py`, `test.py`) are supervised OCR. Legacy self-supervised pretraining code is still present in `pretrain.py`.

## Architecture (Current OCR Path)

- **Backbone**: ViT patch encoder (`models/vit.py`)
- **Recognizer**: CTC classifier head (`TeluguCTCModel` in `models/ocr.py`)
- **Tokenizer**: Grapheme-level Telugu vocab (`vocab.py`)
- **Loss**: `torch.nn.CTCLoss`
- **Decoding**: Greedy or Beam-search (`utils.decode_batch`)

## Dataset Layout

```text
images/                   # all .jpg handwritten word images
labels.csv                # columns: image_id,text
vocab.txt                 # one line: <grapheme>\t<index>
train.txt                 # one line: <image_id> <idx1> <idx2> ...
valid.txt
test.txt
```

### Telugu Text Consistency

- Unicode normalization is applied (**NFC by default**) for vocab entries, labels, and decoded text.
- `image_id` from `labels.csv` is treated as authoritative (`image_id + '.jpg'`).
- ZWNJ is normalized/removed for stable lookup.

## Augmentation & Preprocessing

Training uses:
- small rotation (±5°)
- Gaussian blur
- Gaussian noise
- elastic distortion

Image resizing strategy:
- preserve aspect ratio
- resize by fixed height
- pad width in collate (no width stretching)
- if extremely wide, downscale with aspect ratio preserved

## Install

```bash
pip install torch torchvision pillow pandas numpy tqdm editdistance opencv-python
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
  --num_workers 4 \
  --max_steps_per_epoch 0 \
  --decode_strategy beam \
  --beam_width 10
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
  --decode_strategy beam \
  --beam_width 10 \
  --test_model ./weights/best_telugu_ctc.pt
```

Predictions are written to `pred_logs/test_predictions.tsv`.

## Language Model Hook

Beam decoder accepts an optional LM scoring hook in `utils.decode_batch(...)` (`lm_scorer`, `lm_alpha`) for future Telugu LM integration.
