# Setting up Tennis

The videos include the original videos from [Vid2Player](https://cs.stanford.edu/~haotianz/research/vid2player/) and additional videos.

1. Obtain the videos from YT, following the ids and naming conventions in `data/tennis/videos.csv`.

Make sure to download the correct frame rate and quality settings. We use 1080P for all of the videos. Wimbledon videos are 25.0 FPS and US Open videos are 30 FPS.

2. Run `python3 frames_as_jpg.py tennis <src_video_dir> -o <frame_out_dir>` to extract frames.