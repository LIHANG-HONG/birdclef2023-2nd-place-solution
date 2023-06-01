import argparse
import importlib
from modules.preprocess import prepare_cfg
from modules.model import load_model
import torch
import os
import subprocess

def make_parser():
    parser = argparse.ArgumentParser(description='拡張子変更、アノテーション画像作成')
    parser.add_argument('--model', choices=["sed_v2s",'sed_b3ns','sed_seresnext26t','cnn_v2s','cnn_resnet34d','cnn_b3ns','cnn_b0ns'])
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    model = args.model
    cfg = importlib.import_module(model).basic_cfg
    cfg = prepare_cfg(cfg)

    stage = ''
    model = load_model(cfg,stage,train=False)
    onxx_model_path = os.path.join(cfg.onnx_path,f"{model}.onnx")
    torch.onnx.export(model, cfg.input_shape, onxx_model_path, verbose=True, input_names=cfg.input_names, output_names=cfg.output_names,opset_version=cfg.opt_version)

    proc = subprocess.run(['mo','--input_model', onxx_model_path, 'output_dir', cfg.openvino_path,'--compress_to_fp16'])
    return

if __name__=='__main__':
    main()