import copy
from configs.common import common_cfg
from augmentations import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
)
from audiomentations import Compose as amCompose
from audiomentations import OneOf as amOneOf
from audiomentations import AddBackgroundNoise, Gain, GainTransition, TimeStretch
import numpy as np

cfg = copy.deepcopy(common_cfg)

cfg.model_type = "sed"
cfg.model_name = "tf_efficientnetv2_s_in21k"

cfg.secondary_label = 0.9
cfg.secondary_label_weight = 0.5


cfg.batch_size = 96
cfg.seed = {
    "pretrain_ce": 20231121,
    "pretrain_bce": 20230503,
    "train_ce": 20231019,
    "train_bce": 20231911,
    "finetune": 20230523,
}
cfg.DURATION_TRAIN = 10
cfg.DURATION_FINETUNE = 30
cfg.freeze = False
cfg.mixup = True
cfg.mixup2 = True
cfg.mixup_prob = 0.7
cfg.mixup_double = 0.5
cfg.mixup2_prob = 0.15
cfg.mix_beta = 5
cfg.mix_beta2 = 2
cfg.use_delta = (True,)
cfg.epochs = {
    "pretrain_ce": 70,
    "pretrain_bce": 40,
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
    "pretrain_bce": "outputs/sed_v2s/pretrain_ce/last.ckpt",
    "train_ce": "outputs/sed_v2s/pretrain_bce/last.ckpt",
    "train_bce": "outputs/sed_v2s/train_ce/last.ckpt",
    "finetune": "outputs/sed_v2s/train_bce/last.ckpt",
}

cfg.output_path = {
    "pretrain_ce": "outputs/sed_v2s/pretrain_ce",
    "pretrain_bce": "outputs/sed_v2s/pretrain_bce",
    "train_ce": "outputs/sed_v2s/train_ce",
    "train_bce": "outputs/sed_v2s/train_bce",
    "finetune": "outputs/sed_v2s/finetune",
}

cfg.loss = {
    "pretrain_ce": "ce",
    "pretrain_bce": "bce",
    "train_ce": "ce",
    "train_bce": "bce",
    "finetune": "bce",
}

cfg.img_size = 384
cfg.n_mels = 128
cfg.n_fft = 2048
cfg.f_min = 0
cfg.f_max = 16000

cfg.test_batch_size = int(
    np.max([int(cfg.batch_size / (int(cfg.valid_duration) / cfg.DURATION)), 2])
)
cfg.hop_length = cfg.infer_duration * cfg.SR // (cfg.img_size - 1)
cfg.train_part = int(cfg.DURATION / cfg.infer_duration)
cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)

cfg.normal = 80

cfg.tta_delta = 3

am_audio_transforms = amCompose(
    [
        # sed
        AddBackgroundNoise(
            cfg.birdclef2021_nocall + cfg.birdclef2020_nocall,
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.6,
        ),
        AddBackgroundNoise(
            cfg.freefield + cfg.warblrb + cfg.birdvox,
            min_snr_in_db=0,
            max_snr_in_db=3,
            p=0.3,
        ),
        AddBackgroundNoise(
            cfg.rainforest + cfg.environment, min_snr_in_db=0, max_snr_in_db=3, p=0.4
        ),
        amOneOf(
            [
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
                GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
            ],
        ),
    ]
)


np_audio_transforms = CustomCompose(
    [
        CustomOneOf(
            [
                NoiseInjection(p=1, max_noise_level=0.04),
                GaussianNoise(p=1, min_snr=5, max_snr=20),
                PinkNoise(p=1, min_snr=5, max_snr=20),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
            ],
            p=0.3,
        ),
    ]
)

basic_cfg = cfg
