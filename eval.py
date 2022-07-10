#!/usr/bin/env python3

import os
import argparse
import re

from util.io import load_gz_json, load_json
from util.dataset import DATASETS, FINEGYM_START_SET
from util.score import compute_mAPs
from util.eval import non_maximum_supression


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', help='Path to predictions or model dirs')

    parser.add_argument('-d', '--dataset', type=str, choices=DATASETS)
    parser.add_argument('-s', '--split', type=str, required=True,
                        choices=['train', 'test', 'val'])

    parser.add_argument('--nms_window', type=int, default=1)

    parser.add_argument('-t', '--tolerances', type=int, nargs='+')

    # Start only set for finegym
    parser.add_argument('--start', action='store_true',
                        help='Restrict to start actions only for FineGym')
    return parser.parse_args()


def get_pred_file(pred_dir, split):
    regex = re.compile(r'pred-{}\.(\d+)\.recall\.json\.gz'.format(split))
    candidates = []
    for file_name in os.listdir(pred_dir):
        m = regex.match(file_name)
        if m:
            candidates.append((
                os.path.join(pred_dir, file_name), int(m.group(1))))
    if len(candidates) > 0:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
    raise FileNotFoundError('No suitable prediction file!')


def main(dataset, pred_file, split, nms_window, tolerances, start):
    # Infer the name of the prediction file
    if os.path.isdir(pred_file):
        if dataset is None:
            config = load_json(os.path.join(pred_file, 'config.json'))
            dataset = config['dataset']
            print('Inferred dataset:', dataset)

        if split != 'test':
            _, epoch = get_pred_file(pred_file, 'test')
            pred_file = os.path.join(
                pred_file, 'pred-{}.{}.recall.json.gz'.format(split, epoch))
        else:
            pred_file, _ = get_pred_file(pred_file, split)
        print('Evaluating on: {}'.format(pred_file))
    else:
        assert dataset is not None, 'Dataset is required!'

    pred = (load_gz_json if pred_file.endswith('.gz') else load_json)(
            pred_file)

    truth = load_json(os.path.join('data', dataset, '{}.json'.format(split)))

    if start:
        assert dataset == 'finegym'
        for p in pred:
            p['events'] = [e for e in p['events'] if e['label']
                           in FINEGYM_START_SET]
        for t in truth:
            t['events'] = [e for e in t['events'] if e['label']
                           in FINEGYM_START_SET]
            t['num_events'] = len(t['events'])

    kwargs = {}
    if tolerances is not None:
        kwargs['tolerances'] = tolerances

    print('\n=== Results on {} (w/o NMS) ==='.format(split))
    no_nms_result = compute_mAPs(truth, pred, **kwargs)

    pred = non_maximum_supression(pred, nms_window)
    print('\n=== Results on {} (w/ NMS) ==='.format(split))
    nms_result = compute_mAPs(truth, pred, **kwargs)
    return no_nms_result, nms_result


if __name__ == '__main__':
    main(**vars(get_args()))