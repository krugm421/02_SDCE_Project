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

def undistortImage(rawImage, mtx, dist):
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

def getMatrices(shape = (720, 1280)):
    xsize = shape[1]
    ysize = shape[0]

    # set source and destination points for the perspective transform
    x_magin_bottom = 0.2 * xsize / 2
    x_magin_top = 0.85 * xsize / 2
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
    invTransformationMatrix = cv2.getPerspectiveTransform(dst, src)


    return transformationMatrix, invTransformationMatrix

def perspectiveTransform(imgIn, transformationMatrix):
    xsize = imgIn.shape[1]
    ysize = imgIn.shape[0]

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
        win_xleft_low = np.clip(np.uint(leftx_current - margin), 0, imgIn.shape[1] - 1)  # Update this
        win_xleft_high = np.clip(np.uint(leftx_current + margin), 0, imgIn.shape[1] - 1) # Update this

        win_xright_low = np.clip(np.uint(rightx_current - margin), 0, imgIn.shape[1] - 1)  # Update this
        win_xright_high = np.clip(np.uint(rightx_current + margin), 0, imgIn.shape[1] - 1)  # Update this

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
    ploty = np.linspace(0, imgIn.shape[0] - 1, imgIn.shape[0])
    if (not len(lefty)) <= 0 and (not len(leftx) <= 0):
        left_fit = np.polyfit(lefty, leftx, 2)
        #left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        out_img[lefty, leftx] = [255, 0, 0]
    if (not len(righty)) <= 0 and (not len(rightx) <= 0):
        right_fit = np.polyfit(righty, rightx, 2)
        #right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        out_img[righty, rightx] = [0, 0, 255]

    # Generate an overlay with the fitted ploynoms
    #xdim_overlay = np.int32(max([max(left_fitx), max(right_fitx), imgIn.shape[1]]))
    #ydim_overlay = np.int32(imgIn.shape[0])
    #lineOverlay = np.uint8(np.zeros([imgIn.shape[0], imgIn.shape[1]]))

    '''''''''
    for ii in zip(np.int32(ploty), np.int32(right_fitx)):
        if ii[1] < imgIn.shape[1]:
            out_img[ii] = [255, 255, 0]
    for ii in zip(np.int32(ploty), np.int32(left_fitx)):
        if ii[1] < imgIn.shape[1]:
            out_img[ii] = [255, 255, 0]
    '''''''''

    return left_fit, right_fit, out_img

def drawLane(imgIn, left_fit, right_fit, invTransformationMatrix):

    ploty = np.linspace(0, imgIn.shape[0] - 1, imgIn.shape[0])
    # not empty check!!!
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Generate an overlay with the fitted ploynoms
    xdim_overlay = np.int32(max([max(left_fitx), max(right_fitx), imgIn.shape[1]]))
    ydim_overlay = np.int32(imgIn.shape[0])
    lineMaskWarped = np.uint8(np.zeros([imgIn.shape[0], imgIn.shape[1]]))

    for ii in zip(np.int32(ploty), np.int32(right_fitx)):
        if ii[1] < imgIn.shape[1]:
            #out_img[ii] = [255, 255, 0]
            lineMaskWarped[ii] = 1
    for ii in zip(np.int32(ploty), np.int32(left_fitx)):
        if ii[1] < imgIn.shape[1]:
            #out_img[ii] = [255, 255, 0]
            lineMaskWarped[ii] = 1

    lineMask = perspectiveTransform(lineMaskWarped, invTransformationMatrix)
    # Drwaw lines
    imgIn[np.nonzero(lineMask)] = [255, 255, 0]

    # Fill the lane area
    # Recast the x and y points into usable format for cv2.fillPoly()
    laneMaskWarped = np.uint8(np.zeros([imgIn.shape[0], imgIn.shape[1]]))
    laneMaskWarped = np.zeros_like(imgIn)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(laneMaskWarped, np.int_([pts]), (0,255,0))
    laneMask = perspectiveTransform(laneMaskWarped, invTransformationMatrix)
    #imgIn[np.nonzero(laneMask)] = [0,255,0]

    imgOut = cv2.addWeighted(imgIn, 1, laneMask, 0.3, 0)

    return imgOut

def measureCurvature(ymax, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = 0
    right_curverad = 0
    if not len(left_fit) <= 0:
        #left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        left_curverad = ((1 + (2 * left_fit[0] * ym_per_pix * ymax + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    if not len(right_fit) <= 0:
        #right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * ym_per_pix * ymax + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad

def get_car_x_offset(dim, left_fit, right_fit):
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ymax = dim[0]
    xmax = dim[1]

    if (not len(left_fit) <= 0):
        left_line_point = left_fit[0] * (ymax - 1) ** 2 + left_fit[1] * (ymax - 1) + left_fit[2]

    if (not len(right_fit) <= 0):
        right_line_point = right_fit[0] * (ymax - 1) ** 2 + right_fit[1] * (ymax - 1) + right_fit[2]

    x_offset = xm_per_pix * (((right_line_point + left_line_point) // 2) - (xmax // 2))
    return x_offset

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)


# Set all values which are defined only once as global variables

# Do calibration: Get distortion parameters and camera matrix
mtx, dist = calibrateCam('camera_cal/')
transformationMatrix, invTransformationMatrix = getMatrices()  # GLOBAL!!!!

def processImage(imgIn, isSingleImage=False):
    currentUndisImg = undistortImage(imgIn, mtx, dist)

    # Get gradient binary image
    binaryImg = getBinaryImg(currentUndisImg)

    # Get warped image, ROI is a trapezoid
    warpedBinaryImg = perspectiveTransform(binaryImg, transformationMatrix)

    # Fit lane lines by fitting 2nd oder polynomial
    left_fit, right_fit, highlightedWarpedImg = fitPolynomial(warpedBinaryImg)

    # Get the curvature of each lane line
    left_curverad, right_curverad = measureCurvature(warpedBinaryImg.shape[1], left_fit, right_fit)

    # Get cars x offset
    x_offset = get_car_x_offset(warpedBinaryImg.shape, left_fit, right_fit)

    # Unwarp the line mask we fond
    ImgOut = drawLane(currentUndisImg, left_fit, right_fit, invTransformationMatrix)

    # Print out some parameters and also save all intermediate steps of the pipeline if function used on singel imgs
    if isSingleImage == True:
        print(currentTestImgfile + ': curvature left [m] = ' + str(left_curverad) + '   curvature right [m] = ' + str(
            right_curverad))
        print(currentTestImgfile + ' delta x of car to lane center [m] = ' + str(x_offset))
        cv2.imwrite('output_images/' + 'undis_' + currentTestImgfile, currentUndisImg)
        cv2.imwrite('output_images/' + 'bin_' + currentTestImgfile, binaryImg * 255)
        cv2.imwrite('output_images/' + 'warp_' + currentTestImgfile, warpedBinaryImg * 255)
        cv2.imwrite('output_images/' + 'highlightet_' + currentTestImgfile, highlightedWarpedImg)
        cv2.imwrite('output_images/' + 'out_' + currentTestImgfile, ImgOut)

    return ImgOut

if __name__ == "__main__":

    testImgFiles = os.listdir('test_images/')
    for currentTestImgfile in testImgFiles:
        # check if filetype is jpg
        if not (currentTestImgfile.endswith('jpg')):
            continue

        currentTestImg = cv2.imread('test_images/' + currentTestImgfile)
        processImage(currentTestImg, True)