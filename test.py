import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

from Config import Configs
from loadData import all_data_loader
from models.ocr import TeluguCTCModel
from models.vit import ViT
import utils

sys.stdout.reconfigure(encoding='utf-8')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(cfg, num_classes):
    vit_encoder = ViT(
        image_size=(cfg.img_height, cfg.max_width),
        patch_size=cfg.vit_patch_size,
        num_classes=1000,
        dim=768,
        depth=6,
        heads=8,
        mlp_dim=2048,
    )
    return TeluguCTCModel(vit_encoder=vit_encoder, emb_size=768, num_classes=num_classes)


def main():
    cfg = Configs().parse()
    _, _, test_loader, vocab = all_data_loader(cfg_obj=cfg, batch_size=cfg.batch_size)

    num_classes = vocab.vocab_size + 1
    model = build_model(cfg, num_classes).to(DEVICE)

    if not cfg.test_model:
        raise ValueError('Please provide --test_model checkpoint path')

    ckpt = torch.load(cfg.test_model, map_location=DEVICE)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)

    criterion = nn.CTCLoss(blank=vocab.blank_idx, reduction='mean', zero_infinity=True)
    model.eval()

    predictions, ground_truths = [], []
    total_loss = 0.0

    pred_log = os.path.join('pred_logs', 'test_predictions.tsv')
    if os.path.exists(pred_log):
        os.remove(pred_log)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='test'):
            images = batch['images'].to(DEVICE)
            logits = model(images)
            log_probs = torch.log_softmax(logits, dim=-1)

            input_lengths = torch.full(
                size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long, device=DEVICE
            )
            targets = batch['targets'].to(DEVICE)
            target_lengths = batch['target_lengths'].to(DEVICE)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            preds = utils.decode_batch(log_probs, vocab, strategy=cfg.decode_strategy, beam_width=cfg.beam_width, lm_alpha=cfg.lm_alpha)
            predictions.extend(preds)
            ground_truths.extend(batch['labels'])
            utils.write_ctc_predictions(pred_log, batch['image_ids'], preds)

    cer, wer = utils.compute_word_and_cer(predictions, ground_truths, vocab)
    print(f'Test Loss: {total_loss / max(1, len(test_loader)):.4f}')
    print(f'Test CER: {cer:.4f}')
    print(f'Test WER: {wer:.4f}')


if __name__ == '__main__':
    main()
