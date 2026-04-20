import torch
from torch.utils.data import DataLoader
from functools import partial

from dataset import TeluguOCRDataset, telugu_collate_fn
from vocab import TeluguVocab
from Config import Configs

cfg = Configs().parse()

# Legacy tokens kept for compatibility with pretrain/fine_tune scripts
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens)


def labelDictionary(cfg_obj=None):
    cfg_used = cfg_obj or cfg
    vocab = TeluguVocab(cfg_used.vocab_file, normalization=cfg_used.unicode_norm)
    return vocab.vocab_size, vocab.char2idx, vocab.idx2char


def build_vocab(cfg_obj=None):
    cfg_used = cfg_obj or cfg
    return TeluguVocab(cfg_used.vocab_file, normalization=cfg_used.unicode_norm)


def loadData(cfg_obj=None):
    cfg_used = cfg_obj or cfg
    vocab = build_vocab(cfg_used)
    train_ds = TeluguOCRDataset(
        split_file=cfg_used.train_file,
        image_dir=cfg_used.image_dir,
        label_csv=cfg_used.labels_csv,
        vocab=vocab,
        img_height=cfg_used.img_height,
        max_width=cfg_used.max_width,
        augment=True,
    )
    valid_ds = TeluguOCRDataset(
        split_file=cfg_used.val_file,
        image_dir=cfg_used.image_dir,
        label_csv=cfg_used.labels_csv,
        vocab=vocab,
        img_height=cfg_used.img_height,
        max_width=cfg_used.max_width,
        augment=False,
    )
    test_ds = TeluguOCRDataset(
        split_file=cfg_used.test_file,
        image_dir=cfg_used.image_dir,
        label_csv=cfg_used.labels_csv,
        vocab=vocab,
        img_height=cfg_used.img_height,
        max_width=cfg_used.max_width,
        augment=False,
    )
    return train_ds, valid_ds, test_ds, vocab


def all_data_loader(cfg_obj=None, batch_size=None):
    cfg_used = cfg_obj or cfg
    batch_size = batch_size or cfg_used.batch_size
    train_ds, valid_ds, test_ds, vocab = loadData(cfg_used)
    collate = partial(
        telugu_collate_fn,
        patch_multiple=cfg_used.vit_patch_size,
        max_width=cfg_used.max_width,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg_used.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg_used.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg_used.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, valid_loader, test_loader, vocab
