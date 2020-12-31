import subprocess
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

def concat(dir, output_path='output.mp4'):
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    ds_store = os.path.join(dir, '.DS_Store')
    if ds_store in files:
        files.remove(ds_store)

    files.sort()
    print(files)
    output_file = os.path.join(dir, output_path)

    clips = map(VideoFileClip, files)

    final = concatenate_videoclips(list(clips))
    final.write_videofile(output_file)


