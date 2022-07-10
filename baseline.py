#!/usr/bin/env python3
""" Training for baselines """

import os
import argparse
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from tabulate import tabulate

from model.feature import GRU, TCN, GCN, ASFormer
from model.impl.calf import ContextAwareWeights
from dataset.feature import FeatureDataset
from util.io import store_json, store_gz_json
from util.eval import ForegroundF1, ErrorStat
from util.dataset import DATASETS, load_classes
from util.score import compute_mAPs


EPOCH_NUM_FRAMES = 1000000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('feat_dir', type=str)
    parser.add_argument(
        '-m', '--model_arch', type=str, required=True,
        choices=['gru', 'tcn', 'mstcn', 'gcn', 'asformer'])
    parser.add_argument('--clip_len', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_dir', type=str)
    parser.add_argument('--feat_dims', type=int, nargs=2)

    parser.add_argument('--calf', action='store_true')
    parser.add_argument('--dilate_len', type=int, default=0)

    parser.add_argument('--eval_clip', action='store_true')
    return parser.parse_args()


def evaluate(model, dataset, classes, save_pred=None, clip_len=None):
    classes_inv = {v: k for k, v in classes.items()}

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    for video in dataset.videos:
        feat, label, pad_len = dataset.get(video)
        assert feat.shape[0] == label.shape[0] + 2 * pad_len, (
            feat.shape, label.shape, pad_len)

        if clip_len:
            scores = np.zeros((feat.shape[0], len(classes) + 1))
            support = np.zeros(feat.shape[0], dtype=np.int32)
            for i in range(0, max(1, feat.shape[0] - clip_len // 2 + 1),
                           clip_len // 2):
                tmp = model.predict(feat[i:i + clip_len, :])[1]
                if i + tmp.shape[0] > feat.shape[0]:
                    # Truncate padding
                    tmp = tmp[:feat.shape[0] - i, :]
                scores[i:i + tmp.shape[0], :] += tmp
                support[i:i + tmp.shape[0]] += 1
            assert np.min(support) > 0, (video, support.tolist())
            scores /= support[:, None]
            pred = np.argmax(scores, axis=1)
        else:
            pred, scores = model.predict(feat)

        if pad_len > 0:
            pred = pred[pad_len:-pad_len]
            scores = scores[pad_len:-pad_len]
        assert pred.shape[0] == label.shape[0]
        assert scores.shape[0] == label.shape[0]

        err.update(label, pred)

        events = []
        events_high_recall = []
        for i in range(len(pred)):
            f1.update(label[i], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i, j] >= 0.01:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': scores[i, j].item()
                    })

        pred_events.append({'video': video, 'events': events})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall})

    print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

    def get_f1_tab_row(str_k):
        k = classes[str_k] if str_k != 'any' else None
        return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
    rows = [get_f1_tab_row('any')]
    for c in sorted(classes):
        rows.append(get_f1_tab_row(c))
    print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                   floatfmt='0.2f'))
    print()

    mAPs, _ = compute_mAPs(dataset._labels, pred_events_high_recall)
    print()

    if save_pred is not None:
        store_json(save_pred + '.json', pred_events)
        store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
    return np.mean(mAPs[1:])


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)])


def store_config(file_path, args, num_epochs, classes):
    store_json(file_path, {
        'dataset': args.dataset, 'num_classes': len(classes),
        'clip_len': args.clip_len, 'batch_size': args.batch_size,
        'num_epochs': num_epochs, 'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate, 'eval_clip': args.eval_clip,
        'epoch_num_frames': EPOCH_NUM_FRAMES, 'calf': args.calf,
        'dilate_len': args.dilate_len,
    }, pretty=True)


def build_datasets(args):
    calf_weights = None
    if args.calf:
        calf_weights = ContextAwareWeights()

    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    dataset_len = EPOCH_NUM_FRAMES // args.clip_len
    train_data = FeatureDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.feat_dir, args.clip_len, dataset_len,
        feat_dims=args.feat_dims, calf_weights=calf_weights,
        dilate_len=args.dilate_len)
    train_data.print_info()
    val_data = FeatureDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.feat_dir, args.clip_len, dataset_len // 2,
        feat_dims=args.feat_dims, calf_weights=calf_weights,
        dilate_len=args.dilate_len)
    val_data.print_info()
    return classes, train_data, val_data


def main(args):
    if not os.path.isdir(args.feat_dir):
        args.feat_dir = os.path.join('data', args.dataset, args.feat_dir)

    classes, train_data, val_data = build_datasets(args)
    print('Feature dim:', train_data.feature_dim)

    worker_init_fn = lambda x: random.seed(x + epoch * 10)
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        worker_init_fn=worker_init_fn)

    num_classes = len(classes) + 1
    if args.model_arch == 'gru':
        model = GRU(train_data.feature_dim, num_classes)
    elif args.model_arch == 'tcn':
        model = TCN(train_data.feature_dim, num_classes)
    elif args.model_arch == 'mstcn':
        model = TCN(train_data.feature_dim, num_classes, num_stages=3)
    elif args.model_arch == 'gcn':
        model = GCN(train_data.feature_dim, num_classes)
    elif args.model_arch == 'asformer':
        model = ASFormer(train_data.feature_dim, num_classes)
        args.eval_clip = True
        print('ASFormer requires clip eval due to learned position embedding')
    else:
        raise NotImplementedError()
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    num_steps_per_epoch = len(train_loader)
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    store_config('/dev/stdout', args, num_epochs, classes)

    losses = []
    best_epoch = None
    best_model_dict = None
    best_val_mAP = 0
    for epoch in range(num_epochs):
        train_loss = model.epoch(train_loader, optimizer, scaler,
                                 lr_scheduler=lr_scheduler)
        val_loss = model.epoch(val_loader)
        print('[Epoch {}] Train loss: {:0.3f} Val loss: {:0.3f}'.format(
              epoch, train_loss, val_loss))
        losses.append({'train': train_loss, 'val': val_loss})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses)
            store_config(os.path.join(args.save_dir, 'config.json'),
                         args, num_epochs, classes)

        print('=== Results on VAL (w/o NMS) ===')
        pred_file = None
        if args.save_dir is not None:
            pred_file = os.path.join(args.save_dir, 'pred-val.{}'.format(epoch))
            os.makedirs(args.save_dir, exist_ok=True)
        val_mAP = evaluate(model, val_data, classes, pred_file,
                           args.clip_len if args.eval_clip else None)
        if val_mAP > best_val_mAP:
            best_model_dict = copy.deepcopy(model.state_dict())
            best_val_mAP = val_mAP
            best_epoch = epoch
            if args.save_dir is not None:
                torch.save(best_model_dict,
                           os.path.join(args.save_dir, 'best_epoch.pt'))
            print('New best epoch!')

    print('Best epoch: {}\n'.format(best_epoch))

    del train_data, train_loader, val_data, val_loader
    test_dataset_path = os.path.join('data', args.dataset, 'test.json')
    if best_model_dict is not None and os.path.exists(test_dataset_path):
        model.load(best_model_dict)

        pred_file = None if args.save_dir is None else os.path.join(
            args.save_dir, 'pred-test.{}'.format(best_epoch))
        print('=== Results on TEST (w/o NMS) ===')
        evaluate(model, FeatureDataset(
            classes, test_dataset_path, args.feat_dir, args.clip_len,
            1, feat_dims=args.feat_dims
        ), classes, pred_file, args.clip_len if args.eval_clip else None)


if __name__ == '__main__':
    main(get_args())