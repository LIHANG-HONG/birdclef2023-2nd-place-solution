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
cfg.model_name = "tf_efficientnetv2_s_in21k"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5


cfg.batch_size = 96
cfg.PRECISION = 32
cfg.seed = {
    "pretrain_ce": 20191121,
    "pretrain_bce": 20190503,
    "train_ce": 20191019,
    "train_bce": 2019101911,
    "finetune": 2019101921,
}
cfg.DURATION_TRAIN = 15
cfg.DURATION_FINETUNE = 30
cfg.freeze = False
cfg.mixup = True
cfg.mixup2 = True
cfg.mixup_prob = 0.6
cfg.mixup_double = 0.95
cfg.mixup2_prob = 1.0
cfg.mix_beta = 5
cfg.mix_beta2 = 1
cfg.in_chans = 3
cfg.epochs = {
    "pretrain_ce": 56,
    "pretrain_bce": 54,
    "train_ce": 60,
    "train_bce": 30,
    "finetune": 10,
}
cfg.lr = {
    "pretrain_ce": 3e-4,
    "pretrain_bce": 1e-3,
    "train_ce": 3e-4,
    "train_bce": 1e-3,
    "finetune": 6e-4,
}

cfg.model_ckpt = {
    "pretrain_ce": None,
    "pretrain_bce": "outputs/cnn_v2s/pytorch/pretrain_ce/last.ckpt",
    "train_ce": "outputs/cnn_v2s/pytorch/pretrain_bce/last.ckpt",
    "train_bce": "outputs/cnn_v2s/pytorch/train_ce/last.ckpt",
    "finetune": "outputs/cnn_v2s/pytorch/train_bce/last.ckpt",
}

cfg.output_path = {
    "pretrain_ce": "outputs/cnn_v2s/pytorch/pretrain_ce",
    "pretrain_bce": "outputs/cnn_v2s/pytorch/pretrain_bce",
    "train_ce": "outputs/cnn_v2s/pytorch/train_ce",
    "train_bce": "outputs/cnn_v2s/pytorch/train_bce",
    "finetune": "outputs/cnn_v2s/pytorch/finetune",
}

cfg.final_model_path = "outputs/cnn_v2s/pytorch/finetune/last.ckpt"
cfg.onnx_path = "outputs/cnn_v2s/onnx"
cfg.openvino_path = "outputs/cnn_v2s/openvino"

cfg.loss = {
    "pretrain_ce": "ce",
    "pretrain_bce": "bce",
    "train_ce": "ce",
    "train_bce": "bce",
    "finetune": "bce",
}

cfg.img_size = 500
cfg.n_mels = 64
cfg.n_fft = 1024
cfg.f_min = 50
cfg.f_max = 14000

cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)
cfg.hop_length = 320

cfg.normal = 80

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
