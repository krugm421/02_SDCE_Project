import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

DEBUG_FLAG = True


def debugPrint(str):
    if DEBUG_FLAG == True:
        print(str)


def calibrateCam(imgPath, nx=9, ny=6):
    CalibrationImageFiles = os.listdir(imgPath)

    # set up the word point of the chessboard
    # objpoints = np.array([ii, jj, 0] for jj in range(ny) for ii in range(nx), )
    objpointsRange = np.array([[[ii, jj, 0]] for jj in range(ny) for ii in range(nx)], np.float32)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for currentFile in CalibrationImageFiles:
        # check if filetype if jpg
        if not currentFile.endswith('jpg'):
            continue
        # read image and get grayscale image
        currentImageIn = cv2.imread(imgPath + currentFile)
        gray = cv2.cvtColor(currentImageIn, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # add corners and world points and save image with highlighted corners:
        if ret == True:
            objpoints.append(objpointsRange)
            imgpoints.append(corners)

            # currentImageOut = cv2.drawChessboardCorners(currentImageIn, (nx, ny), corners, ret)
            # cv2.imwrite('output_images/' + 'output_' + currentFile, currentImageOut)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    debugPrint('Camera matrix: ' + str(mtx))
    debugPrint('Distorion parameters: ' + str(dist))

    return mtx, dist


def undistorImage(rawImage, mtx, dist):
    undistImg = currentImageOut = cv2.undistort(rawImage, mtx, dist, None, mtx)
    return undistImg

def getBinaryImg(imgIn):
    threshold_sobel_min = 30
    threshold_sobel_max = 100

    threshold_saturation_min = 150
    threshold_saturation_max = 255

    # Do color transform and get S channel
    s_channel = cv2.cvtColor(imgIn, cv2.COLOR_BGR2HLS)[:, :, 2]

    # apply thresholds to saturation channel
    mask_saturation = np.zeros_like(s_channel)
    mask_saturation[(s_channel > threshold_saturation_min) & (s_channel <= threshold_saturation_max)] = 1

    # get derivative in x direction
    gray = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    soble_x_scaled = np.uint8(255 * np.abs(sobel_x) / np.max(sobel_x))

    # apply threshold to gradient
    mask_sobel = np.zeros_like(soble_x_scaled)
    mask_sobel[(soble_x_scaled > threshold_sobel_min) & (soble_x_scaled <= threshold_sobel_max)] = 1

    # merge both masks
    mask_merged = np.zeros_like(mask_sobel)
    mask_merged[(mask_sobel == 1) | (mask_saturation == 1)] = 1

    return mask_merged

def perspectiveTransform(imgIn):
    xsize = imgIn.shape[1]
    ysize = imgIn.shape[0]

    # set source and destination points for the perspective transform
    x_magin_bottom = 0.2 * xsize/2
    x_magin_top = 0.8 * xsize/2
    y_hight = 0.6 * ysize
    src = np.array([
        [0, ysize - 1],
        [xsize - 1, ysize - 1],
        [xsize - x_magin_top, y_hight],
        [x_magin_top, y_hight]], dtype="float32")

    dst = np.array([
        [0, ysize - 1],
        [xsize - 1, ysize - 1],
        [xsize - 1, 0],
        [0, 0]], dtype="float32")

    # Get transform matrix and warp image
    transformationMatrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(imgIn, transformationMatrix, (xsize, ysize), flags=cv2.INTER_LINEAR)

    return warped

def findLanePixels(imgIn):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(imgIn[imgIn.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((imgIn, imgIn, imgIn))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(imgIn.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = imgIn.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = imgIn.shape[0] - (window + 1) * window_height
        win_y_high = imgIn.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = np.uint(leftx_current - margin)  # Update this
        win_xleft_high = np.uint(leftx_current + margin)  # Update this

        win_xright_low = np.uint(rightx_current - margin)  # Update this
        win_xright_high = np.uint(rightx_current + margin)  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        mask_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)) \
                         & ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high))
        good_left_inds = mask_left_inds.nonzero()[0]

        mask_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high)) \
                          & ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high))
        good_right_inds = mask_right_inds.nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If  found > minpix pixels, recenter next window ###
        if len(good_left_inds) > minpix:
            # recenter left box
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            # recenter right box
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fitPolynomial(imgIn):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = findLanePixels(imgIn)

    # Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, imgIn.shape[0] - 1, imgIn.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    for ii in range(len(ploty)):
        out_img[np.int32(ploty), np.int32(right_fitx), :] = [255, 255, 0]
        out_img[np.int32(ploty), np.int32(left_fitx), :] = [255, 255, 0]

    return out_img

if __name__ == "__main__":

    # Do calibration: Get distortion parameters and camera matrix
    mtx, dist = calibrateCam('camera_cal/')

    testImgFiles = os.listdir('test_images/')
    for currentTestImgfile in testImgFiles:
        # check if filetype is jpg
        if not (currentTestImgfile.endswith('jpg')):
            continue

        currentTestImg = cv2.imread('test_images/' + currentTestImgfile)
        currentUndisImg = undistorImage(currentTestImg, mtx, dist)

        # Get gradient binary image
        binaryImg = getBinaryImg(currentUndisImg)

        # Get warped image, ROI is a trapezoid
        warpedBinaryImg = perspectiveTransform(binaryImg)

        # Fit lane lines
        highlightedImg = fitPolynomial(warpedBinaryImg)

        cv2.imwrite('output_images/' + 'undis_' + currentTestImgfile, currentUndisImg)
        cv2.imwrite('output_images/' + 'bin_' + currentTestImgfile, binaryImg * 255)
        cv2.imwrite('output_images/' + 'warp_' + currentTestImgfile, warpedBinaryImg * 255)
        cv2.imwrite('output_images/' + 'highlightet_' + currentTestImgfile, highlightedImg)