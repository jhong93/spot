import os
import random
import numpy as np
from torch.utils.data import Dataset

from util.io import load_json


# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5


def _load_features(feat_dir, labels, feat_dims):
    feature_dim = None
    features = {}
    for v in labels:
        video = v['video']
        video_nobs = video.replace('/', '__')
        f = np.load(os.path.join(feat_dir, '{}.npy'.format(video_nobs)),
                    mmap_mode='r')
        if feat_dims is not None:
            if len(f.shape) == 3:
                f = f[:, :, feat_dims[0]:feat_dims[1]]
            else:
                f = f[:, feat_dims[0]:feat_dims[1]]
        features[video] = f

        if feature_dim is None:
            feature_dim = f.shape[-1]
        else:
            assert feature_dim == f.shape[-1]
    return features, feature_dim


class FeatureDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            feat_dir,
            clip_len,
            dataset_len,
            feat_dims=None,
            calf_weights=None,
            dilate_len=0,
            pad_len=DEFAULT_PAD_LEN
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}

        num_frames = [v['num_frames'] for v in self._labels]
        self._weights_by_length = \
            np.array(num_frames, dtype=np.float) / np.sum(num_frames)

        self._features, self._feature_dim = _load_features(
            feat_dir, self._labels, feat_dims)

        self._clip_len = clip_len
        self._dataset_len = dataset_len
        self._dilate_len = dilate_len
        self._calf_weights = calf_weights
        self._pad_len = pad_len

        assert not (self._calf_weights is not None and self._dilate_len > 0), \
            'Dilation and CALF not implemented'

    @property
    def feature_dim(self):
        return self._feature_dim

    def __getitem__(self, unused):
        video_meta = random.choices(
            self._labels, weights=self._weights_by_length)[0]
        video_feat = self._features[video_meta['video']]
        video_len = video_meta['num_frames']
        if video_len > self._clip_len + 1:
            base_idx = random.randint(-self._pad_len, video_len - self._clip_len - 1 + self._pad_len)
        else:
            base_idx = self._pad_len

        if len(video_feat.shape) == 3:
            version_idx = random.randint(0, video_feat.shape[1] - 1)
            video_feat = video_feat[:, version_idx, :]

        if self._calf_weights is not None:
            calf = np.ones((self._clip_len, len(self._class_dict), 3),
                            dtype=np.float)

        labels = np.zeros(self._clip_len, dtype=np.int64)
        for event in video_meta['events']:
            event_frame = event['frame']

            label_idx = event_frame - base_idx
            if (label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event['label']]
                for i in range(
                        max(0, label_idx - self._dilate_len),
                        min(self._clip_len,
                            label_idx + self._dilate_len + 1)
                ):
                    labels[i] = label

                if self._calf_weights is not None:
                    w = self._calf_weights.weights
                    ofs = self._calf_weights.offset

                    # Truncate end of w, if past end
                    end_overflow = (label_idx + w.shape[0] - ofs
                                    - self._clip_len)
                    if end_overflow > 0:
                        w = w[:-end_overflow, :]

                    # Skip beginning of w, if past start
                    begin_overflow = ofs - label_idx
                    if begin_overflow > 0:
                        w = w[begin_overflow:, :]
                        ofs -= begin_overflow
                    calf[label_idx - ofs:label_idx - ofs + w.shape[0],
                         label - 1, :] = w

        pos_base_idx = max(0, base_idx)
        feat = video_feat[pos_base_idx:pos_base_idx + self._clip_len, :].copy()

        mask = np.ones(self._clip_len, dtype=np.bool)
        mask[feat.shape[0]:] = 0

        if feat.shape[0] < self._clip_len:
            pad_start = 0 if base_idx > 0 else -base_idx
            pad_end = self._clip_len - feat.shape[0] - pad_start
            feat = np.pad(feat, ((pad_start, pad_end), (0, 0)))
        ret = {'feature': feat, 'mask': mask, 'label': labels,
                'contains_event': int(np.sum(labels) > 0)}
        if self._calf_weights is not None:
            ret['calf'] = calf
        return ret

    def __len__(self):
        return self._dataset_len

    def get(self, video):
        feat = self._features[video]
        if len(feat.shape) == 3:
            feat = feat[:, 0, :]    # 0th channel is unaugmented
        feat = feat.copy()          # Make copy since array is memmapped

        labels = np.zeros(feat.shape[0], np.int)
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta['num_frames']
        for event in meta['events']:
            frame = event['frame']
            if frame < num_frames:
                labels[frame] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))

        # Pad start and end of the sequence
        if self._pad_len > 0:
            feat = np.pad(feat, ((self._pad_len, self._pad_len), (0, 0)))
        return feat, labels, self._pad_len

    @property
    def videos(self):
        return sorted(self._features.keys())

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames,
            num_events / num_frames * 100))
