# Spotting Temporally Precise, Fine-Grained Events in Video

This repository contains code for our paper:

*Spotting Temporally Precise, Fine-Grained Events in Video*\
In ECCV 2022\
James Hong, Haotian Zhang, Michael Gharbi, Matthew Fisher, Kayvon Fatahalian

Link: [project website](https://jhong93.github.io/projects/spot.html)

```
@inproceedings{precisespotting_eccv22,
    author={Hong, James and Zhang, Haotian and Gharbi, Micha\"{e}l and Fisher, Matthew and Fatahalian, Kayvon},
    title={Spotting Temporally Precise, Fine-Grained Events in Video},
    booktitle={ECCV},
    year={2022}
}
```

This code is released under the BSD-3 [LICENSE](/LICENSE).

## Overview

Our paper presents a study of temporal event detection (spotting) in video at the precision of a single or small (e.g., 1-2) tolerance of frames.
This is a useful task for annotating video for analysis and synthesis when temporal precision is important, and we demonstrate fine-grained events in several sports as examples.

In this regime, the most crucial design aspect is end-to-end learning of spatial-temporal features from the pixels.
We present a surprisingly strong, compact, and end-to-end learned baseline that is conceptually simpler than the two-phase architectures common in the temporal action detection, segmentation, and spotting literature.

## Environment

The code is tested in Linux (Ubuntu 16.04 and 20.04) with the dependency versions in ```requirements.txt```. Similar environments are likely to work also but YMMV.

## Datasets

Refer to the READMEs in the [data](/data) directory for pre-processing and setup instructions.

## Basic usage

To train a model, use `python3 train_e2e.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch>`.

* `<dataset_name>`: supports tennis, fs_comp, fs_perf, finediving, finegym, soccernetv2, soccernet_ball
* `<frame_dir>`: path to the extracted frames
* `<save_dir>`: path to save logs, checkpoints, and inference results
* `<model_arch>`: feature extractor architecture (e.g., RegNet-Y 200MF w/ GSM : `rny002_gsm`)

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

To evaluate a set of predictions with the mean-AP metric, use `python3 eval.py -s <split> <model_dir_or_prediction_file>`.
* `<model_dir_or_prediction_file>`: can be the saved directory of a model containing predictions or a path to a prediction file.

The predictions are saved as either `pred-{split}.{epoch}.recall.json.gz` or `pred-{split}.{epoch}.json` files. The latter contains only the top class predictions for each frame, omitting all background, while the former contains all non-background detections above a low threshold, to complete the precision-recall curve.

We also save per-frame scores, `pred-{split}.{epoch}.score.json.gz`, which can be used to combine predictions from multiple models (see `eval_ensemble.py`).

### Trained models

Models and configurations can be found at https://github.com/jhong93/e2e-spot-models/. Place the checkpoint file and config.json file in the same directory.

To perform inference with an already trained model, use `python3 test_e2e.py <model_dir> <frame_dir> -s <split> --save`. This will save the predictions in the model directory, using the default file naming scheme.

### Baselines

Implementations for several baselines in the paper are in `baseline,py`. TSP and 2D-VPD features are available in our [Google Drive](https://drive.google.com/drive/folders/1AQFd8JsvxdEG2jQfY5GDVSLEtc9r824W?usp=sharing).

## Using your own data

Each dataset has plaintext files that contain the list of classes and events in each video.

#### class.txt

This is a list of the class names, one per line.

#### {split}.json

This file contains entries for each video and its contained events.

```
[
    {
        "video": VIDEO_ID,
        "num_frames": 4325,                 // Video length
        "num_events": 10,
        "events": [
            {
                "frame": 525,               // Frame
                "label": CLASS_NAME,        // Event class
                "comment": ""               // Optional comments
            },
            ...
        ],
        "fps": 25,
        "width": 1920,      // Metadata about the source video
        "height": 1080
    },
    ...
]
```

#### Frame directory

We assume pre-extracted frames (either RGB in jpg format or optical flow), that have been resized to 224 pixels high or similar. The organization of the frames is expected to be `<frame_dir>/<video_id>/<frame_number>.jpg`. For example,

```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```

#### Prediction file format

Predictions are formatted similarly to the labels:
```
[
    {
        "video": VIDEO_ID,
        "events": [
            {
                "frame": 525,               // Frame
                "label": CLASS_NAME,        // Event class
                "score": 0.96142578125
            },
            ...
        ],
        "fps": 25           // Metadata about the source video
    },
    ...
]
```