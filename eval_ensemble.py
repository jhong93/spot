#!/usr/bin/env python3
""" Naively average prediction scores from multiple models """

import os
import argparse
import re
import numpy as np

from util.io import load_json, load_gz_json, store_json, store_gz_json
from util.dataset import DATASETS, FINEGYM_START_SET, load_classes
from util.score import compute_mAPs
from util.eval import non_maximum_supression


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('file_names', type=str, nargs='+')
    parser.add_argument('-s', '--split', type=str, required=True,
                        choices=['train', 'test', 'val'])
    parser.add_argument('--nms_window', type=int, default=1)

    # Start only set for finegym
    parser.add_argument('--start', action='store_true',
                        help='Restrict to start actions only for FineGym')

    parser.add_argument('-o', '--out_dir', type=str)
    return parser.parse_args()


def get_score_file(pred_dir, split):
    regex = re.compile(r'pred-{}\.(\d+)\.score\.json'.format(split))
    for file_name in os.listdir(pred_dir):
        m = regex.match(file_name)
        if m:
            return os.path.join(pred_dir, file_name), int(m.group(1))
    raise FileNotFoundError('No suitable prediction file!')


def ensemble(dataset, results, fps_dict=None):
    classes = load_classes(os.path.join('data', dataset, 'class.txt'))
    classes_inv = {v: k for k, v in classes.items()}

    pred_events = []
    pred_events_high_recall = []
    for video in sorted(results[0].keys()):
        scores = None
        for r in results:
            r_scores = np.array(r[video])
            if scores is None:
                scores = np.zeros_like(r_scores)
            scores += r_scores / len(results)
        pred = np.argmax(scores, axis=1)

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):
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

        pred_video = {'video': video, 'num_events': len(events),
                      'events': events}
        pred_video_hr = {'video': video, 'num_events': len(events_high_recall),
                         'events': events_high_recall}
        if fps_dict is not None:
            pred_video['fps'] = fps_dict[video]
            pred_video_hr['fps'] = fps_dict[video]
        pred_events.append(pred_video)
        pred_events_high_recall.append(pred_video_hr)
    return pred_events, pred_events_high_recall


def main(dataset, file_names, split, out_dir, start, nms_window):
    results = []
    for file_name in file_names:
        # Infer the name of the prediction file
        if os.path.isdir(file_name):
            if split != 'test':
                _, epoch = get_score_file(file_name, 'test')
                file_name = os.path.join(
                    file_name, 'score-{}.{}.score.json.gz'.format(split, epoch))
            else:
                file_name, _ = get_score_file(file_name, split)
            print('Evaluating on: {}'.format(file_name))

        results.append(load_gz_json(file_name))

    pred_events, pred_events_high_recall = ensemble(dataset, results)

    truth = load_json(os.path.join('data', dataset, '{}.json'.format(split)))

    if start:
        assert dataset == 'finegym'
        for p in pred_events:
            p['events'] = [e for e in p['events'] if e['label']
                           in FINEGYM_START_SET]
        for p in pred_events_high_recall:
            p['events'] = [e for e in p['events'] if e['label']
                           in FINEGYM_START_SET]
        for t in truth:
            t['events'] = [e for e in t['events'] if e['label']
                           in FINEGYM_START_SET]
            t['num_events'] = len(t['events'])

    print('\n=== Results on {} (w/o NMS) ==='.format(split))
    compute_mAPs(truth, pred_events_high_recall, plot_pr=False)

    print('\n=== Results on {} (w/ NMS) ==='.format(split))
    compute_mAPs(truth, non_maximum_supression(pred_events_high_recall,
                 nms_window), plot_pr=False)

    if out_dir is not None:
        os.makedirs(out_dir)
        store_json(os.path.join(out_dir, 'pred-{}.0.json'.format(split)),
                   pred_events)
        store_gz_json(
            os.path.join(out_dir, 'pred-{}.0.recall.json.gz'.format(split)),
            pred_events_high_recall)
        print('Saved predictions: {}'.format(out_dir))


if __name__ == '__main__':
    main(**vars(get_args()))