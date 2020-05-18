import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip

import AdvancedLaneFinding

# Perform calibration
mtx, dist = AdvancedLaneFinding.calibrateCam('camera_cal/')

def processImage(imgIn):


if __name__ == '__main__':



    # Load
    videoFile = "project_video.mp4"
    vid = VideoFileClip(videoFile)
