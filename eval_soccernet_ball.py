#!/usr/bin/env python3
""" Wrapper around SoccerNet scoring API """

import os
import argparse
from collections import defaultdict
import tempfile
from tabulate import tabulate

from util.io import load_json, load_gz_json, store_json
from util.eval import non_maximum_supression
from eval import get_pred_file
from eval_ensemble import ensemble

from SoccerNet.Evaluation.ActionSpotting import evaluate as sn_evaluate
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_BALL


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', nargs='+', type=str,
                        help='Path(s) to soccernet predictions or model dirs')
    parser.add_argument('-s', '--split', type=str, required=True,
                        choices=['train', 'val', 'test', 'challenge'])
    parser.add_argument('--nms_window', type=int, default=25)
    parser.add_argument('-l' , '--soccernet_path', required=True,
                        help='Path to SoccerNet-ball labels')
    parser.add_argument('--eval_dir',
                        help='Path to save intermediate files to. E.g., for sumbission to the evaluation server.')
    return parser.parse_args()


def store_eval_files(raw_pred, eval_dir):
    game_pred = defaultdict(list)
    for obj in raw_pred:
        game, half = obj['video'].rsplit('/', 1)
        half = int(half)
        for event in obj['events']:
            ss = event['frame'] / obj['fps']
            position = int(ss * 1000)

            mm = int(ss / 60)
            ss = int(ss - mm * 60)
            game_pred[game].append({
                'gameTime': '{} - {}:{:02d}'.format(half, mm, ss),
                'label': event['label'],
                'half': str(half),
                'position': str(position),
                'confidence': str(event['score'])
            })

    for game, pred in game_pred.items():
        game_out_dir = os.path.join(eval_dir, game)
        os.makedirs(game_out_dir)
        store_json(os.path.join(game_out_dir, 'results_spotting.json'), {
            'UrlLocal': game, 'predictions': pred
        }, pretty=True)


def load_fps_dict(ref_file):
    return {v['video']: v['fps'] for v in load_gz_json(ref_file)}


def main(pred_file, split, soccernet_path, nms_window, eval_dir):
    if len(pred_file) == 1:
        pred_file = pred_file[0]
        if os.path.isdir(pred_file):
            pred_file, _ = get_pred_file(pred_file, split)
            #assert split != 'challenge', \
            #    'Do not infer pred file if challenge due to bad data!'
            print('Evaluating on: {}'.format(pred_file))
        pred = (load_gz_json if pred_file.endswith('.gz') else load_json)(
                pred_file)
    else:
        scores = []
        fps_dict = None
        for p in pred_file:
            if os.path.isdir(p):
                assert split != 'challenge', \
                    'Do not infer pred file if challenge due to bad data!'
                p2, epoch = get_pred_file(p, split)
                print('Evaluating on: {}'.format(p))
                if fps_dict is None:
                    fps_dict = load_fps_dict(p2)
                scores.append(load_gz_json(os.path.join(
                    p, 'pred-{}.{}.score.json.gz'.format(split, epoch))))
            else:
                if fps_dict is None:
                    fps_dict = load_fps_dict(p.replace('score', 'recall'))
                scores.append(load_gz_json(p))
        _, pred = ensemble('soccernet_ball', scores, fps_dict=fps_dict)

    if nms_window > 0:
        print('Applying NMS:', nms_window)
        pred = non_maximum_supression(pred, nms_window)

    if eval_dir is None:
        tmp_eval_dir = tempfile.TemporaryDirectory(prefix='soccernet-ball-eval')
        eval_dir = tmp_eval_dir.name
    store_eval_files(pred, eval_dir)

    print('Done processing prediction files!')

    split_name = split
    if split == 'val':
        split_name = 'valid'

    def eval_wrapper(metric):
        results = sn_evaluate(
            SoccerNet_path=soccernet_path, Predictions_path=eval_dir,
            split=split_name, version=2, metric=metric, framerate=25, label_files="Labels-ball.json", num_classes=2, dataset="Ball")

        rows = []
        for i in range(len(results['a_mAP_per_class'])):
            label = INVERSE_EVENT_DICTIONARY_BALL[i]
            rows.append((
                label,
                '{:0.2f}'.format(results['a_mAP_per_class'][i] * 100),
                '{:0.2f}'.format(results['a_mAP_per_class_visible'][i] * 100),
                '{:0.2f}'.format(results['a_mAP_per_class_unshown'][i] * 100)
            ))
        rows.append((
            'Average mAP',
            '{:0.2f}'.format(results['a_mAP'] * 100),
            '{:0.2f}'.format(results['a_mAP_visible'] * 100),
            '{:0.2f}'.format(results['a_mAP_unshown'] * 100)
        ))

        print('Metric:', metric)
        print(tabulate(rows, headers=['', 'Any', 'Visible', 'Unseen']))

    eval_wrapper('loose')
    eval_wrapper('tight')
    eval_wrapper('at1')
    eval_wrapper('at2')
    eval_wrapper('at3')
    eval_wrapper('at4')
    eval_wrapper('at5')


if __name__ == '__main__':
    main(**vars(get_args()))