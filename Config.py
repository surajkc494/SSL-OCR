import argparse


class Configs:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Dataset / Telugu OCR config
        self.parser.add_argument('--language', type=str, default='telugu')
        self.parser.add_argument('--charset', type=str, default='telugu_graphemes')
        self.parser.add_argument('--data_path', type=str, default='./data/', help='Dataset root path')
        self.parser.add_argument('--image_dir', type=str, default='./images/', help='Path with .jpg word images')
        self.parser.add_argument('--labels_csv', type=str, default='labels.csv')
        self.parser.add_argument('--vocab_file', type=str, default='vocab.txt')
        self.parser.add_argument('--train_file', type=str, default='train.txt')
        self.parser.add_argument('--val_file', type=str, default='valid.txt')
        self.parser.add_argument('--test_file', type=str, default='test.txt')

        self.parser.add_argument('--img_height', type=int, default=64)
        self.parser.add_argument('--img_width', type=int, default=512)
        self.parser.add_argument('--max_width', type=int, default=512)
        self.parser.add_argument('--vit_patch_size', type=int, default=16)
        self.parser.add_argument('--max_text_len', type=int, default=64)

        # Training / runtime
        self.parser.add_argument('--train_type', type=str, default='normal', choices=['normal', 'stn', 'htr_Augm'])
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--lr', type=float, default=1.5e-4)
        self.parser.add_argument('--weights_path', type=str, default='./weights/')
        self.parser.add_argument('--test_model', type=str, default='')
        self.parser.add_argument('--pretrained_encoder_path', type=str, default='')

        # Auto-computed at runtime from vocab
        self.parser.add_argument('--num_classes', type=int, default=1246)
        self.parser.add_argument('--blank_index', type=int, default=1245)

        # Pre-train legacy switches
        self.parser.add_argument('--vis_results', type=bool, default=True)

    def parse(self):
        return self.parser.parse_args()
