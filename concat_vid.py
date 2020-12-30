import subprocess
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

path = '/Volumes/nathanbackup/fourier/vids/test_10'
files = [os.path.join(path, file) for file in os.listdir(path)]
files.remove(os.path.join(path, '.DS_Store'))
files.sort()
print(files)
output_file = os.path.join(path, 'output.mp4')

clips = []
for file in files:
    clips.append(VideoFileClip(file))

final = concatenate_videoclips(clips)
final.write_videofile(output_file)
