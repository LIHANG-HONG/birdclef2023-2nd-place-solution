# birdclef2023-2nd-place-solution

Writeup for this solution can be found on [kaggle](https://www.kaggle.com/competitions/birdclef-2023/discussion/412707).

## Environment

Google Colaboratory Pro+ with A100 GPU.

Run the following command to install dependencies.

```sh
pip3 install -r requirements.txt
```

## Data Preparation

### train metadata, audios

Metadata(train.csv) can be downloaded from [here](https://www.kaggle.com/datasets/honglihang/birdclef2023-extended-train) and put it to ./inputs

Download the audios from [BirdCLEF2023](https://www.kaggle.com/competitions/birdclef-2023/data), [BirdCLEF2022](https://www.kaggle.com/competitions/birdclef-2022/data), [BirdCLEF2021](https://www.kaggle.com/competitions/birdclef-2021/data) and [extended BirdCLEF2020](https://www.kaggle.com/competitions/birdclef-2023/discussion/398318).

<strong>For additional audios with 2023 species, metadata of audios is included in ./inputs/train.csv. Use the "file" column in csv to download audios from [Xeno-canto](https://xeno-canto.org) and modify the audios to mono channel, 32kHz, ogg format. Some of them may become unavailable because the uploader may have deleted the audio from the site. You can also download the newly uploaded audios not included in train.csv and enhance the train.csv.</strong>

Put all the audios to ./inputs/train_audios.

### background_noise

Download the audios from [here](https://www.kaggle.com/datasets/honglihang/background-noise) and put all the audios to ./inputs/background_noise

## Train

Before running the training, modify configs/common.py, write your WANDB_API_KEY to cfg.WANDB_API_KEY.

Training can be initialized with:

```sh
# if you want to use pseudo as target label, specify the option --use_pseudo
# pseudo label doesn't improve the score and is a vulnerable point in training pipeline
python3 train.py --stage STAGE --model_name MODEL_NAME
```

After training, the last checkpoint (model weights) will be saved to the folder ./outputs/MODEL_NAME/pytorch/STAGE

## Convert model

Run

```sh
python3 convert.py --model_name MODEL_NAME
```

The onnx model will be saved to the folder ./outputs/MODEL_NAME/onnx.

The openvino model will be saved to the folder ./outputs/MODEL_NAME/openvino.

## Inference

Inference is published in a kaggle kernel [here](https://www.kaggle.com/code/honglihang/2nd-place-solution-inference-kernel). Weights from our trained models are provided in a kaggle dataset linked to the inference kernel [here](https://www.kaggle.com/datasets/honglihang/birdclef-openvino-comp).
