import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Config import Configs
from loadData import all_data_loader
from models.ocr import TeluguCTCModel
from models.vit import ViT
import utils


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


def run_eval(model, loader, criterion, vocab):
    model.eval()
    losses = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='valid', leave=False):
            images = batch['images'].to(DEVICE)
            logits = model(images)
            log_probs = torch.log_softmax(logits, dim=-1)

            input_lengths = torch.full(
                size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long, device=DEVICE
            )
            targets = batch['targets'].to(DEVICE)
            target_lengths = batch['target_lengths'].to(DEVICE)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            losses += loss.item()

            preds = utils.ctc_greedy_decode_batch(log_probs, vocab)
            all_preds.extend(preds)
            all_gts.extend(batch['labels'])

    cer, wer = utils.compute_word_and_cer(all_preds, all_gts, vocab)
    return losses / max(1, len(loader)), cer, wer


def main():
    cfg = Configs().parse()
    train_loader, valid_loader, _, vocab = all_data_loader(cfg.batch_size)

    utils.validate_split_file(cfg.train_file, vocab)
    utils.validate_split_file(cfg.val_file, vocab)

    cfg.num_classes = vocab.vocab_size + 1
    cfg.blank_index = vocab.blank_idx

    model = build_model(cfg, cfg.num_classes).to(DEVICE)
    criterion = nn.CTCLoss(blank=vocab.blank_idx, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    best_cer = float('inf')
    os.makedirs(cfg.weights_path, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for batch in tqdm(train_loader, desc=f'train epoch {epoch}', leave=False):
            images = batch['images'].to(DEVICE)
            logits = model(images)
            log_probs = torch.log_softmax(logits, dim=-1)

            input_lengths = torch.full(
                size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long, device=DEVICE
            )
            targets = batch['targets'].to(DEVICE)
            target_lengths = batch['target_lengths'].to(DEVICE)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        val_loss, val_cer, val_wer = run_eval(model, valid_loader, criterion, vocab)
        print(
            f'Epoch {epoch}: train_loss={train_loss:.4f} '
            f'val_loss={val_loss:.4f} CER={val_cer:.4f} WER={val_wer:.4f}'
        )

        if val_cer <= best_cer:
            best_cer = val_cer
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'num_classes': cfg.num_classes,
                    'blank_index': cfg.blank_index,
                },
                os.path.join(cfg.weights_path, 'best_telugu_ctc.pt'),
            )


if __name__ == '__main__':
    main()
