import cv2
from typing import NamedTuple


class VideoMetadata(NamedTuple):
    fps: float
    num_frames: int
    width: int
    height: int


def _get_metadata(vc):
    fps = vc.get(cv2.CAP_PROP_FPS)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMetadata(fps, num_frames, width, height)


def get_metadata(video_path):
    vc = cv2.VideoCapture(video_path)
    try:
        return _get_metadata(vc)
    finally:
        vc.release()


def get_frame(video_file, frame_num, height=0):
    vc = cv2.VideoCapture(video_file)
    try:
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = vc.read()
        if height > 0:
            h, w, _ = frame.shape
            width = int(w * height / h)
            frame = cv2.resize(frame, (width, height))
    finally:
        vc.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def cut_segment_cv2(video_file, out_file, start, end):
    print('Extracting using cv2:', out_file)
    vc = cv2.VideoCapture(video_file)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = vc.get(cv2.CAP_PROP_FPS)

    vo = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MP4V'),
                         fps, (width, height))
    vc.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start):
        ret, frame = vc.read()
        assert ret
        vo.write(frame)

    vc.release()
    vo.release()