import argparse
import importlib

def make_parser():
    parser = argparse.ArgumentParser(description='拡張子変更、アノテーション画像作成')
    parser.add_argument('--stage', choices=["pretrain_ce","pretrain_bce","train_ce","train_bce","finetune"])
    parser.add_argument('--model', choices=["sed_v2s",'sed_b3ns','sed_seresnext26t','cnn_v2s','cnn_resnet34d','cnn_b3ns','cnn_b0ns'])
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    stage = args.stage
    model = args.model
    cfg = importlib.import_module(model).basic_cfg
