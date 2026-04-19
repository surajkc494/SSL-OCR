import os
import sys
from typing import List

import editdistance
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')


def validate_split_file(split_file, vocab):
    """Check all indices in split file are within [0, vocab_size-1]."""
    with open(split_file, encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            for tok in parts[1:]:
                idx = int(tok)
                if idx < 0 or idx >= vocab.vocab_size:
                    raise ValueError(
                        f'Out-of-range index in {split_file}:{line_no}: {idx} not in [0, {vocab.vocab_size - 1}]'
                    )


def compute_word_and_cer(predictions: List[str], ground_truths: List[str], vocab):
    total_cer = 0.0
    total_words = len(ground_truths)
    for pred, gt in zip(predictions, ground_truths):
        try:
            pred_tokens = vocab.encode(pred)
        except ValueError:
            pred_tokens = []
        gt_tokens = vocab.encode(gt)
        total_cer += editdistance.eval(pred_tokens, gt_tokens) / max(1, len(gt_tokens))

    wer = sum(pred != gt for pred, gt in zip(predictions, ground_truths)) / max(1, total_words)
    cer = total_cer / max(1, total_words)
    return cer, wer


def ctc_greedy_decode_batch(log_probs, vocab):
    pred_ids = torch.argmax(log_probs, dim=2)
    pred_ids = pred_ids.transpose(0, 1).detach().cpu().tolist()
    return [vocab.ctc_decode(seq) for seq in pred_ids]


def convert_image_np(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(transformer, loader, device):
    with torch.no_grad():
        data = next(iter(loader))['images'].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = transformer.transformer.stn(data).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def write_ctc_predictions(path, image_ids, predictions):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for image_id, pred in zip(image_ids, predictions):
            f.write(f'{image_id}\t{pred}\n')
