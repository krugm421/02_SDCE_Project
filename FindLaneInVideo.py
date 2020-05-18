import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip

import AdvancedLaneFinding



if __name__ == '__main__':

    # Load video
    videoFile = "project_video.mp4"
    vid = VideoFileClip(videoFile)
    white_clip = vid.fl_image(AdvancedLaneFinding.processImage)  # NOTE: this function expects color images!!
    white_clip.write_videofile("out_" + videoFile, audio=False)


