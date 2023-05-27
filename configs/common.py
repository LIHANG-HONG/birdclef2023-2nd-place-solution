from types import SimpleNamespace
import torch

cfg = SimpleNamespace(**{})

cfg.infer_duration = 5
cfg.valid_duration = 60
cfg.label_smoothing = 0.1
cfg.weight_decay = 1e-3
cfg.SR = 32000
cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

common_cfg = cfg