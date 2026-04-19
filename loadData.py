import torch
from torch.utils.data import DataLoader

from dataset import TeluguOCRDataset, telugu_collate_fn
from vocab import TeluguVocab
from Config import Configs

cfg = Configs().parse()

# Legacy tokens kept for compatibility with pretrain/fine_tune scripts
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens)


def labelDictionary():
    vocab = TeluguVocab(cfg.vocab_file)
    return vocab.vocab_size, vocab.char2idx, vocab.idx2char


def build_vocab():
    return TeluguVocab(cfg.vocab_file)


def loadData():
    vocab = build_vocab()
    train_ds = TeluguOCRDataset(
        split_file=cfg.train_file,
        image_dir=cfg.image_dir,
        label_csv=cfg.labels_csv,
        vocab=vocab,
        img_height=cfg.img_height,
        max_width=cfg.max_width,
        augment=True,
    )
    valid_ds = TeluguOCRDataset(
        split_file=cfg.val_file,
        image_dir=cfg.image_dir,
        label_csv=cfg.labels_csv,
        vocab=vocab,
        img_height=cfg.img_height,
        max_width=cfg.max_width,
        augment=False,
    )
    test_ds = TeluguOCRDataset(
        split_file=cfg.test_file,
        image_dir=cfg.image_dir,
        label_csv=cfg.labels_csv,
        vocab=vocab,
        img_height=cfg.img_height,
        max_width=cfg.max_width,
        augment=False,
    )
    return train_ds, valid_ds, test_ds, vocab


def all_data_loader(batch_size=None):
    batch_size = batch_size or cfg.batch_size
    train_ds, valid_ds, test_ds, vocab = loadData()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=telugu_collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=telugu_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=telugu_collate_fn,
    )
    return train_loader, valid_loader, test_loader, vocab
