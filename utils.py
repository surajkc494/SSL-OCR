import math
import os
import sys
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

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
        pred_norm = vocab.normalize_text(pred)
        gt_norm = vocab.normalize_text(gt)
        pred_tokens = vocab.encode(pred_norm, strict=False)
        gt_tokens = vocab.encode(gt_norm, strict=False)
        total_cer += editdistance.eval(pred_tokens, gt_tokens) / max(1, len(gt_tokens))

    wer = sum(vocab.normalize_text(pred) != vocab.normalize_text(gt) for pred, gt in zip(predictions, ground_truths))
    wer /= max(1, total_words)
    cer = total_cer / max(1, total_words)
    return cer, wer


def ctc_greedy_decode_batch(log_probs, vocab):
    pred_ids = torch.argmax(log_probs, dim=2)
    pred_ids = pred_ids.transpose(0, 1).detach().cpu().tolist()
    return [vocab.ctc_decode(seq) for seq in pred_ids]


def _beam_search_decode_single(
    log_probs: torch.Tensor,
    vocab,
    beam_width: int = 10,
    lm_scorer: Optional[Callable[[Sequence[int]], float]] = None,
    lm_alpha: float = 0.0,
) -> str:
    """
    Approximate CTC beam search over token IDs.
    log_probs: [T, C]
    """
    time_steps, num_classes = log_probs.shape
    beams: List[Tuple[Tuple[int, ...], float]] = [(tuple(), 0.0)]

    for t in range(time_steps):
        step_scores = log_probs[t]
        topk_scores, topk_idx = torch.topk(step_scores, k=min(num_classes, beam_width * 3))

        candidates = defaultdict(lambda: -math.inf)
        for prefix, prefix_score in beams:
            for cls, cls_score in zip(topk_idx.tolist(), topk_scores.tolist()):
                new_prefix = prefix + (cls,)
                score = prefix_score + cls_score

                if lm_scorer is not None and lm_alpha > 0.0:
                    score += lm_alpha * lm_scorer(new_prefix)

                if score > candidates[new_prefix]:
                    candidates[new_prefix] = score

        beams = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:beam_width]

    best_path = list(beams[0][0]) if beams else []
    return vocab.ctc_decode(best_path)


def ctc_beam_decode_batch(
    log_probs,
    vocab,
    beam_width: int = 10,
    lm_scorer: Optional[Callable[[Sequence[int]], float]] = None,
    lm_alpha: float = 0.0,
):
    batch = log_probs.transpose(0, 1).detach().cpu()  # [B, T, C]
    preds = []
    for sample in batch:
        preds.append(
            _beam_search_decode_single(
                sample,
                vocab=vocab,
                beam_width=beam_width,
                lm_scorer=lm_scorer,
                lm_alpha=lm_alpha,
            )
        )
    return preds


def decode_batch(log_probs, vocab, strategy='greedy', beam_width=10, lm_scorer=None, lm_alpha=0.0):
    if strategy == 'beam':
        return ctc_beam_decode_batch(
            log_probs,
            vocab,
            beam_width=beam_width,
            lm_scorer=lm_scorer,
            lm_alpha=lm_alpha,
        )
    return ctc_greedy_decode_batch(log_probs, vocab)


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

        _, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def write_ctc_predictions(path, image_ids, predictions):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for image_id, pred in zip(image_ids, predictions):
            f.write(f'{image_id}\t{pred}\n')
