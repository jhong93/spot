# Setting up FS-Competition split

The videos are the same as those from earlier work, VPD: https://jhong93.github.io/projects/vpd.html.

1. Obtain the videos from YT, following the ids and naming conventions in `data/fs_comp/videos.csv`.

Make sure to download the correct frame rate (We used ID 248, VP9, 1920x1080, 25 FPS, in mkv).

2. Run `python3 frames_as_jpg.py fs <src_video_dir> -o <frame_out_dir>` to extract frames.
