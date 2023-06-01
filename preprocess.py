import pandas as pd
import numpy as np
from ast import literal_eval
import torch
import os

def prepare_cfg(cfg,stage):
    if stage in ["pretrain_ce","pretrain_bce"]:
        cfg.bird_cols = cfg.bird_cols_pretrain
    elif stage in ["train_ce","train_bce","finetune"]:
        cfg.bird_cols = cfg.bird_cols_train
    else:
        raise NotImplementedError

    if stage == 'finetune':
        cfg.DURATION = cfg.DURATION_FINETUNE
        cfg.freeze = True
    elif stage in ["pretrain_ce","pretrain_bce","train_ce","train_bce"]:
        cfg.DURATION = cfg.DURATION_TRAIN
    else:
        raise NotImplementedError

def preprocess(cfg):
    def transforms(audio):
        audio = cfg.np_audio_transforms(audio)
        audio = cfg.am_audio_transforms(audio,sample_rate=cfg.SR)
        return audio

    df = pd.read_csv(cfg.train_data)
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: literal_eval(x))
    df['secondary_labels_2023'] = df['secondary_labels_2023'].apply(lambda x: literal_eval(x))
    df['secondary_labels_strict'] = df['secondary_labels_strict'].apply(lambda x: literal_eval(x))
    df['secondary_labels_very_strict'] = df['secondary_labels_very_strict'].apply(lambda x: literal_eval(x))
    df['version'] = df['version'].astype(str)
    df['rating'] = df['rating'].mask(np.isnan(df['rating'].values),df['q'].map({'A':5,'B':4,'C':3,'D':2,'E':1,'no score':0}))
    df['filename'] = df['id'].apply(lambda x: f'XC{x}.ogg')
    df['path'] = df['id'].apply(lambda x: os.path.join(cfg.train_dir,f'XC{x}.ogg'))
    # ensure all the train data is available
    assert df['path'].apply(lambda x:os.path.exists(x)).all()

    labels = np.zeros(shape=(len(df),len(cfg.bird_cols)))
    df_labels = pd.DataFrame(labels,columns=cfg.bird_cols)
    class_sample_count = {col:0 for col in cfg.bird_cols}
    include_in_train = []
    presence_type = []
    for i,(primary_label, secondary_labels) in enumerate(zip(df[cfg.primary_label_col].values,df[cfg.secondary_labels_col].values)):
        include = False
        presence = 'background' if primary_label!='soundscape' else 'soundscape'
        if primary_label in cfg.bird_cols:
            include = True
            presence = 'foreground'
            df_labels.loc[i,primary_label] = 1
            class_sample_count[primary_label] += 1
        for secondary_label in secondary_labels:
            if secondary_label in cfg.bird_cols:
                include = True
                df_labels.loc[i,secondary_label] = cfg.secondary_label
                class_sample_count[secondary_label] += cfg.secondary_label_weight
        presence_type.append(presence)
        include_in_train.append(include)

    df['presence_type'] = presence_type
    df = df[include_in_train].reset_index(drop=True)
    df_labels = df_labels[include_in_train].reset_index(drop=True)

    df_labels = df_labels[(df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground')].reset_index(drop=True)
    df = df[(df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground')].reset_index(drop=True)

    sample_weight = np.zeros(shape=(len(df,)))
    for i,(primary_label, secondary_labels) in enumerate(zip(df[cfg.primary_label_col].values,df[cfg.secondary_labels_col].values)):
        if primary_label in cfg.bird_cols:
            sample_weight[i] = 1.0/(class_sample_count[primary_label])
        else:
            secondary_labels_include = [secondary_label for secondary_label in secondary_labels if secondary_label in cfg.bird_cols]
            sample_weight[i] = np.mean([1.0/class_sample_count[secondary_label] for secondary_label in secondary_labels_include])

    sample_weight = torch.from_numpy(sample_weight)
    return df, df_labels, sample_weight , transforms