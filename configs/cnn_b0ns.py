import copy
from configs.common import common_cfg
from modules.augmentations import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    GaussianNoiseSNR,
    PinkNoiseSNR,
)
from audiomentations import Compose as amCompose
from audiomentations import OneOf as amOneOf
from audiomentations import AddBackgroundNoise, Gain, GainTransition, TimeStretch
import numpy as np

cfg = copy.deepcopy(common_cfg)
if cfg.WANDB_API_KEY=='your key':
    print('input your wandb api key!')
    raise NotImplementedError

cfg.model_type = "cnn"
cfg.model_name = "tf_efficientnet_b0_ns"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5


cfg.batch_size = 156
cfg.PRECISION = 32
cfg.seed = {
    #"pretrain_ce": 20231121,
    #"pretrain_bce": 20230503,
    "train_ce": 202111210524,
    #"train_bce": 20231911,
    #"finetune": 20230523,
}
cfg.DURATION_TRAIN = 20
cfg.DURATION_FINETUNE = 30
cfg.freeze = False
cfg.mixup = False
cfg.mixup2 = True
cfg.mixup_prob = 0.3
cfg.mixup_double = 1.0
cfg.mixup2_prob = 1.0
cfg.mix_beta = 5
cfg.mix_beta2 = 1
cfg.in_chans = 1
cfg.epochs = {
    #"pretrain_ce": 70,
    #"pretrain_bce": 40,
    "train_ce": 50,
    #"train_bce": 30,
    #"finetune": 10,
}
cfg.lr = {
    #"pretrain_ce": 3e-4,
    #"pretrain_bce": 1e-3,
    "train_ce": 3e-4,
    #"train_bce": 1e-3,
    #"finetune": 6e-4,
}

cfg.model_ckpt = {
    #"pretrain_ce": None,
    #"pretrain_bce": "outputs/cnn_b0ns/pytorch/pretrain_ce/last.ckpt",
    "train_ce": None,
    #"train_bce": "outputs/cnn_b0ns/pytorch/train_ce/last.ckpt",
    #"finetune": "outputs/cnn_b0ns/pytorch/train_bce/last.ckpt",
}

cfg.output_path = {
    #"pretrain_ce": "outputs/cnn_b0ns/pytorch/pretrain_ce",
    #"pretrain_bce": "outputs/cnn_b0ns/pytorch/pretrain_bce",
    "train_ce": "outputs/cnn_b0ns/pytorch/train_ce",
    #"train_bce": "outputs/cnn_b0ns/pytorch/train_bce",
    #"finetune": "outputs/cnn_b0ns/pytorch/finetune",
}

cfg.final_model_path = "outputs/cnn_b0ns/pytorch/train_ce/last.ckpt"
cfg.onnx_path = "outputs/cnn_b0ns/onnx"
cfg.openvino_path = "outputs/cnn_b0ns/openvino"

cfg.loss = {
    #"pretrain_ce": "ce",
    #"pretrain_bce": "bce",
    "train_ce": "ce",
    #"train_bce": "bce",
    #"finetune": "bce",
}

cfg.img_size = 256
cfg.n_mels = 128
cfg.n_fft = 2048
cfg.f_min = 0
cfg.f_max = 16000

cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)
cfg.hop_length = cfg.infer_duration * cfg.SR // (cfg.img_size - 1)

cfg.normal = 255

cfg.am_audio_transforms = amCompose([
    AddBackgroundNoise(cfg.birdclef2021_nocall + cfg.birdclef2020_nocall + cfg.freefield + cfg.warblrb + cfg.birdvox + cfg.rainforest + cfg.environment, min_snr_in_db=3.0,max_snr_in_db=30.0,p=0.5),
    Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.2),

])


cfg.np_audio_transforms = CustomCompose([
  CustomOneOf([
    NoiseInjection(p=0.5, max_noise_level=0.04),
    GaussianNoiseSNR(p=0.5),
    PinkNoiseSNR(p=0.5)
  ]),
])

cfg.input_shape = (120,cfg.in_chans,cfg.n_mels,cfg.img_size)
cfg.input_names = [ "x" ]
cfg.output_names = [ "y" ]
cfg.opset_version = 10

basic_cfg = cfg
