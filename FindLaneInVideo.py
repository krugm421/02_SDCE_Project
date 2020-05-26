import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip
import queue

import AdvancedLaneFinding

# Define a class to receive the characteristics of each line detection
class LaneFinder():
    def __init__(self):
        # was the lane detected in the last iteration?
        self.Detected = False
        # x values of the last n fits of the line
        #self.recent_xfitted_left = []
        #self.recent_xfitted_right = []
        #average x values of the fitted line over the last n iterations
        #self.bestx_left = None
        #self.bestx_right = None
        #polynomial coefficients over the last n iterations
        self.queueLength = 5
        self.last_fits_left = queue.Queue(self.queueLength)
        self.last_fits_right = queue.Queue(self.queueLength)
        #polynomial coefficients for the most recent fit
        self.current_fit_left = [np.array([False])]
        self.current_fit_right = [np.array([False])]
        #radius of curvature of the line in some units
        #self.radius_of_curvature_left = None
        #self.radius_of_curvature_right = None
        #distance in meters of vehicle center from the line
        #self.line_base_pos_left = None
        #self.line_base_pos_right = None
        #difference in fit coefficients between last and new fits
        #self.diffs_left = np.array([0,0,0], dtype='float')
        #self.diffs_right = np.array([0, 0, 0], dtype='float')
        #x values for detected line pixels
        #self.allx_left = None
        #self.allx_right = None
        #y values for detected line pixels
        #self.ally_left = None
        #self.ally_right = None

        self.mtx, self.dist = AdvancedLaneFinding.calibrateCam('camera_cal/')
        self.transformationMatrix, self.invTransformationMatrix = AdvancedLaneFinding.getMatrices()

    def update(self, imgIn):
        #undistort image
        imgUndis = AdvancedLaneFinding.undistortImage(imgIn, self.mtx, self.dist)

        # Get warped image, ROI is a trapezoid
        warpedImg = AdvancedLaneFinding.perspectiveTransform(imgUndis, self.transformationMatrix)
        # Get gradient binary image
        warpedBinaryImg = AdvancedLaneFinding.getBinaryImg(warpedImg)
        # Find line pixels


        if self.Detected == False:
            # If did not find a line before use sliding window
            leftx, lefty, rightx, righty, warpedHighlightedBinaryImg  = AdvancedLaneFinding.findLinePixelsSlidingWindow(warpedBinaryImg)
            self.Detected = True
        else:
            leftx, lefty, rightx, righty, warpedHighlightedBinaryImg = AdvancedLaneFinding.findLinePixelsPreviousFit(warpedBinaryImg, self.current_fit_left, self.current_fit_right)
        # Fit the polynom
        fit_left, fit_right = AdvancedLaneFinding.fitPolynomial(imgIn.shape, leftx, lefty, rightx, righty)
        if len(fit_left) <= 0 or len(fit_right) <= 0:
            fit_left = self.current_fit_left
            fit_right = self.current_fit_right
        else:
            self.current_fit_left = fit_left
            self.current_fit_right = fit_right

        # get mean fit over the last frames
        if self.last_fits_left.qsize() != 0:
            fit_left_mean = np.mean(np.array(self.last_fits_left.queue), 0)
        else:
            fit_left_mean = fit_left
        if self.last_fits_right.qsize() != 0:
            fit_right_mean = np.mean(np.array(self.last_fits_right.queue), 0)
        else:
            fit_right_mean = fit_right
        self.current_fit_left = fit_left_mean
        self.current_fit_right = fit_right_mean

        left_curvature_current, right_curvature_current = AdvancedLaneFinding.measureCurvature(imgIn.shape[0], fit_left, fit_right)
        left_curvature_mean, right_curvature_mean = AdvancedLaneFinding.measureCurvature(imgIn.shape[0], fit_left_mean, fit_right_mean)
        deltax_current = AdvancedLaneFinding.getCarX_offset(imgIn.shape, fit_left, fit_right)
        deltax_mean = AdvancedLaneFinding.getCarX_offset(imgIn.shape, fit_left_mean, fit_right_mean)

        sanityCheck = True
        #sanityCheck = sanityCheck and (np.abs(deltax_mean - deltax_current)/np.abs(deltax_mean) <= 0.25)

        # are both fits almost parallel?
        p = (fit_left[1] - fit_right[1])/(fit_left[0] - fit_right[0])
        q = (fit_left[2] - fit_right[2])/(fit_left[0] - fit_right[0])

        y1 = -p/2 + np.sqrt((p/2) ** 2 - q)
        y2 = -p/2 - np.sqrt((p/2) ** 2 - q)

        sanityCheck = sanityCheck and not (0 <= y1 < imgIn.shape[0]) and not (0 <= y2 < imgIn.shape[0])

        if sanityCheck:
            self.last_fits_left.put(fit_left)
            self.last_fits_right.put(fit_right)
        else:
            # emty queue
            self.last_fits_right = queue.Queue(self.queueLength)
            self.last_fits_right = queue.Queue(self.queueLength)
            self.Detected = False

        # Visialize found lane
        imgOut = AdvancedLaneFinding.drawLane(imgUndis, fit_left_mean, fit_right_mean, self.invTransformationMatrix)

        cv2.putText(imgOut, "L. Curvature: %.2f km" % (left_curvature_mean / 1000), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(imgOut, "R. Curvature: %.2f km" % (right_curvature_mean / 1000), (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(imgOut, "Offset to lane center: %.2f m" % deltax_mean, (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 255, 255), 2)
        # cv2.putText(imgOut, "X2: %.2f" % y2, (50, 140), cv2.FONT_HERSHEY_DUPLEX,
        #             1,
        #             (255, 255, 255), 2)
        # cv2.putText(imgOut, "a_0: %.6f" % fit_left[0], (50, 170), cv2.FONT_HERSHEY_DUPLEX,
        #             1,
        #             (255, 255, 255), 2)
        # cv2.putText(imgOut, "b_0: %.6f" % fit_right[0], (50, 200), cv2.FONT_HERSHEY_DUPLEX,
        #             1,
        #             (255, 255, 255), 2)
        cv2.putText(imgOut, "Sanity Check: %i" % sanityCheck, (50, 230), cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (255, 255, 255), 2)



        # update queue
        if self.last_fits_left.full():
            self.last_fits_left.get()
        if self.last_fits_right.full():
            self.last_fits_right.get()

        #return cv2.cvtColor(warpedBinaryImg*255, cv2.COLOR_GRAY2BGR)
        #return warpedHighlightedBinaryImg
        return  imgOut

if __name__ == '__main__':

    laneFinderObj = LaneFinder()

    # Load video
    videoFile = "project_video.mp4"
    vid = VideoFileClip(videoFile)
    white_clip = vid.fl_image(laneFinderObj.update)  # NOTE: this function expects color images!!
    white_clip.write_videofile("out_" + videoFile, audio=False)


