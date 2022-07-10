#!/usr/bin/env python3
""" Visualize predictions in the browser """

import os
import argparse
import re
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, send_file, jsonify

from util.eval import non_maximum_supression
from util.io import load_json, load_gz_json
from util.video import get_frame
from util.dataset import DATASETS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('pred_dir', help='Path to the model files')
    parser.add_argument('frame_dir', help='Path to extracted frames')
    parser.add_argument('-v', '--video_dir', help='Path to the video files')
    parser.add_argument('-f', '--flow_dir', help='Path to extracted flow')

    # Network settings
    parser.add_argument('-p', '--port', type=int, default=8000)
    parser.add_argument('--public', action='store_true',
                        help='Serve on all IPs. NOT SECURE!')

    parser.add_argument('--nms', action='store_true')
    return parser.parse_args()


PRED_FILE_RE = re.compile(r'^pred-\w+\.\d+\.json$')


def list_pred_files(pred_dir):
    ret = []
    for file_name in sorted(os.listdir(pred_dir)):
        if PRED_FILE_RE.match(file_name):
            ret.append(file_name.split('.json')[0])
    return ret


def build_app(dataset, pred_dir, video_dir, frame_dir, flow_dir, nms):
    data_dir = os.path.join('data', dataset)
    train = load_json(os.path.join(data_dir, 'train.json'))
    val = load_json(os.path.join(data_dir, 'val.json'))
    test = []
    try:
        test = load_json(os.path.join(data_dir, 'test.json'))
    except Exception as e:
        print(e)

    challenge = []
    try:
        challenge = load_json(os.path.join(data_dir, 'challenge.json'))
    except Exception as e:
        print(e)

    labels = train + val + test + challenge
    class_names = list(sorted(
        {e['label'] for x in labels for e in x['events']}))

    print('Loading frames from:', frame_dir, flow_dir)

    def get_video_path(v):
        return os.path.join(video_dir, '{}.mp4'.format(v))

    app = Flask(__name__, template_folder='web/templates',
                static_folder='web/static')

    pred_files = list_pred_files(pred_dir)
    videos = ([('train', v['video']) for v in train]
                + [('val', v['video']) for v in val]
                + [('test', v['video']) for v in test]
                + [('challenge', v['video']) for v in challenge])
    videos.sort()

    @app.route('/')
    def root():
        return render_template('index.html', pred_files=pred_files,
                               pred_dir=pred_dir, labels=class_names,
                               videos=videos)

    @app.route('/labels.json')
    def get_labels():
        return jsonify(labels)

    @app.route('/pred/<pred_file>')
    def get_pred(pred_file):
        # pred_path = os.path.join(pred_dir, '{}.json'.format(pred_file))
        # pred = load_json(pred_path)
        pred_path = os.path.join(
            pred_dir, '{}.recall.json.gz'.format(pred_file))
        pred = load_gz_json(pred_path)
        if nms:
            pred = non_maximum_supression(pred, 1)
            for v in pred:
                v['events'] = [e for e in v['events'] if e['score'] >= 0.2]
                v['num_events'] = len(v['events'])
        return jsonify(pred)

    @app.route('/rgb/<video_name>/<int:frame_num>')
    def get_rgb_frame(video_name, frame_num):
        frame_path = os.path.join(
            frame_dir, video_name.replace('=', '/'),
            '{:06d}.jpg'.format(frame_num))
        return send_file(frame_path, mimetype='image/jpeg', conditional=True)

    @app.route('/flow/<video_name>/<int:frame_num>')
    def get_flow_frame(video_name, frame_num):
        if flow_dir is not None:
            frame_path = os.path.join(
                flow_dir, video_name.replace('=', '/'),
                '{:06d}.jpg'.format(frame_num))
            return send_file(frame_path, mimetype='image/jpeg',
                             conditional=True)
        else:
            # Null image
            im = Image.new('RGB', (16, 9))
            fp = BytesIO()
            im.save(fp, format='jpeg')
            fp.seek(0)
            return send_file(fp, mimetype='image/jpeg')

    def get_video_path(v):
        return os.path.join(video_dir, '{}.mp4'.format(v))

    @app.route('/full_res/<video_name>/<int:frame_num>')
    def get_full_res(video_name, frame_num):
        if video_dir is not None:
            frame = get_frame(get_video_path(video_name), frame_num)
            im = Image.fromarray(frame)
            fp = BytesIO()
            im.save(fp, format='jpeg')
            fp.seek(0)
            return send_file(fp, mimetype='image/jpeg')
        else:
            return jsonify(['No video dir specified. Unable to decode high res frames.'])

    return app


def main(args, debug=True):
    app = build_app(args.dataset, args.pred_dir, args.video_dir, args.frame_dir,
                    args.flow_dir, args.nms)
    app_kwargs = {'debug': debug, 'port': args.port}
    if args.public:
        app_kwargs['host'] = '0.0.0.0'
    app.run(**app_kwargs)


if __name__ == '__main__':
    main(get_args())
