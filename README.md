## Advanced Lane Finding Project


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[disChessboard]: ./output_images/out_dis_calibration3.jpg
[undisChessboard]: ./output_images/out_undis_calibration3.jpg
[undisRoad]: ./output_images/undis_test2.jpg
[binary]: ./output_images/bin_test2.jpg
[warped2]: ./output_images/birdseye_test2.jpg
[warped3]: ./output_images/birdseye_test3.jpg
[warped4]: ./output_images/birdseye_test4.jpg
[warped5]: ./output_images/birdseye_test5.jpg
[warped6]: ./output_images/birdseye_test6.jpg
[highlightedBinary]: ./output_images/highlightet_test2.jpg
[result]: ./output_images/out_test2.jpg
[video]: ./out_project_video.mp4 "Video"
[AdvancedLaneFinding.py]: ./AdvancedLaneFinding.py
[FindLaneInVideo.py]:  ./FindLaneInVideo.py
[resultFolder]: ./output_images 

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points


---

### Summary


This project involved the application of some advanced computer vision techniques for lane line finding.

The code is split up in to two files:
1. [AdvancedLaneFinding.py] --> This sourcefile contains all individual steps involved for camera calibration, lane detection, visualisation and also the code for lane finding for the the individual test images. The result images of each step are in the folder [resultFolder].
2. [FindLaneInVideo.py] --> Contains the application of the steps to the test videos. In addition a class 'LaneFinder' is defined here which responsible for handling the lane detection over several consecutive frame of the video. This includes smoothing the lane lines fit and a sanity check which detects unplausible results of the lane finding. Included is the result for the default test [video]. 



### Camera Calibration


The function `calibrateCam(imgPath, nx=9, ny=6)` performs the calibration of the camera with help of the available chessboard images. 

For each corner of the 9x6 chessboards world points are generated. The coordinates are normalized to the length of one chessboard square (e.g the top left corner gets the coordinates x=0, y=0, the next to the left gets y=1, y=0). As all corner points are in the x/y plane of the chessboard the z coordinate will be 0 for all chessboard images in different poses. 

The image coordinates of each corner in each chessboard image are detected with the `ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)` function. 

With `ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)` the intrinsic camera matrix and also the distortion parameters are estimated based on the pairs of word and and image point coordinates of the corner we found earlier.

A visual inspection of the original and undistorted images shows that the 'barrel' distortion which is prevalent in the test images is removed after undistoring with the found parameters.  
![alt text][disChessboard]      
![alt text][undisChessboard]

The found parameters will be used later on to undistort the images and also the video used in the lane finding pipeline.

### Pipeline (single images)

Each provided test image is processed with the pipeline to detect the current lane. Each step is represented by functions which will be discussed in detail. 

All functions are defined in [AdvancedLaneFinding.py]. The processing of each image takes place in the ``if __name__ == "__main__":`` section. The pipeline can be run just by running [AdvancedLaneFinding.py].

#### 1. Distortion correction

The found distortion parameters from the previous calibration step will we used to perform the undistortion of the test images for lane finding. This is important as an undistorted image is necessary to compensate the cameras distortion in every image to improve the accuracy of the estimate of the lane shape.
![alt text][undisRoad]

#### 2. Line pixel extraction 
A combination of color and gradient thresholds are used to extract the line pixels. This step is performed in the function `getBinaryImg(imgIn)`.

1. The color threshold is based on an the images representation in HLS (= hue, lightness and saturation) color space. We make use of the saturation channel. As in the HLS color space lightness and saturation are two independent values it's especially useful to handle changing lighting conditions e.g. shadows. 

2. For the gradient a sobel x gradient filter is used. The sobel x filter extracts gradients in x direction. The use of a sobel x filter is reasonable as the projections of the lines in the image will always be relatively steep in the image which will case a high gradient in x direction.

Both binary images are combined with a pixel-wise or operation.
![alt text][binary]

#### 3. Perspective transform

A perspective transform is now used to warp the image to a top view on the lane. 

First the transformation matrix between the original an warped perspective must be calculated. Is is done in the `getMatrices()` function which returns the transformation matrix and also it's inverse for forward and backward transformation. To do that we must chose four source and four destination points. The source or region of interest (=ROI) was chosen as a trapezoid. It is important to get a good compromise. We want to be able to detect curved lines which necessitates the angles of non-parallel sides to be wide enough. In the same way the angles shouldn't be to wide to prevent e.g. falsely detect road limits or neighboring lines.  

The source points where tuned that all of the test images lines, especially for images with high curvature lines, are within the ROI with a reasonable margin. 
```python
    x_magin_top = 0.82 * xsize / 2
    y_hight = 0.65 * ysize
    src = np.array([
        [0, ysize - 1],
        [xsize - 1, ysize - 1],
        [xsize - x_magin_top, y_hight],
        [x_magin_top, y_hight]], dtype="float32")
```
The destination points are chosen that the ROI will fill up the whole warped image.
```python
    dst = np.array([
        [0, ysize - 1],
        [xsize - 1, ysize - 1],
        [xsize - 1, 0],
        [0, 0]], dtype="float32")
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 719      | 0, 719        | 
| 1279, 719      | 1279, 719      |
| 756, 468     | 1279, 0      |
| 525, 468      | 0, 0        |

The function facilitates `transformationMatrix = cv2.getPerspectiveTransform(src, dst)` to get the the transformation matrix and also `invTransformationMatrix = cv2.getPerspectiveTransform(dst, src)` to get the inverse transformation matrix.

With the resulting transformation matrix we can perform the forward transformation with `warped = cv2.warpPerspective(imgIn, transformationMatrix, (xsize, ysize), flags=cv2.INTER_LINEAR)`. A visual check shows that all lanes are covered by the top view and are relatively parallel.
![alt text][warped2]
![alt text][warped3]
![alt text][warped4]
![alt text][warped5]
![alt text][warped6]


#### 4. Line pixel detection

The detection of the line-pixels takes place in the function `findLinePixelsSlidingWindow(imgIn)`. The finding is base on the binary image which we warped to the top view perspective. 

The algorithm used for pixel selection is based on the sliding window method. 

The staring x coordinates for the sliding windows is calculated using a histogram over all y columns. The maximum on the left and right half of the histogram will be used as initial x coordinates.

Each pixel which is within a box centered around the the maximums of the histogram will be counted as a line pixel. The mean x coordinate will be used to center the following box.  

The detected pixels will now be used to fit the line. The shape which is used is a 2nd order polynomial. The polynomial is fitted in the `fitPolynomial(shape, leftx, lefty, rightx, righty)` function using the `np.polyfit(lefty, leftx, 2)` of numpy.

![alt text][highlightedBinary]

#### 5. Curvature radius and car position

Based on the two polynomial the curvature radius of both lines and also the cars position relative to the found lines are calculated.

The curvature calculation is handled in `measureCurvature(ymax, left_fit, right_fit)`. The function makes use of the formula from the lecture. 

Also, the cars offset in respect to the found lane lines is calculated in `getCarX_offset(dim, left_fit, right_fit)`. This function calculates the difference of the mean between the bottom-most x-coordinates of both lines to the center of the image. This assumes that the camera is centered on the car.

#### 6. Visualising the result

With the fitted polynomials and the measurements we have everything to display the final result.

The highlighting of the found lane is done in ``drawLane(imgIn, left_fit, right_fit, invTransformationMatrix)``. First a binary image is calculated which selects all pixels between the two polynomials. This image is transformed back to the original perspective using the inverse transformation matrix. The binary image is now used to index each lane pixel in the original undistorted image and highlight it using open CVs `cv2.fillPoly(laneMaskWarped, np.int_([pts]), (0,255,255))` function.

Also, the curvature and lane center offset are displayed. 

![alt text][result]

---

### Pipeline (video)

The implementation of the pipeline to process the project video can be found in [FindLaneInVideo.py]. All the previously discussed functions within the pipeline will be reused for the processing of the video. To better handle the results over several steps I implemented the `LaneFinder` class. The steps performed for each frame are defined by it's `update(self, imgIn)` method. The pipeline involves the same steps as for the single image processing. Similar to the single image pipeline one has to run [FindLaneInVideo.py] to start the pipeline for the video.

The `LaneFinder` class stores the last n fitted polynomial coefficients in a FIFO queue. This queue is used to average the found fits over the last frames to mitigate the jitter caused e.g. by dashed lines or rough road surfaces. I chose a queue length of 5. The averaging is done by calculating the mean value of each coefficient and using this mean fit to visualise.

Also a simple sanity check is performed to detect fits which are not roughly parallel. This check if the two polynomials are valid:
1. The two fits are not allowed to intersect within the warped image. I calculate the y coordinates where both curves (if at all) intersect. If those points are on the image the sanity check fails.
2. Both lines should bend in the same direction. This can be checked easily by checking if the coefficients of y^2 for both lines have a different signs. The check is only performed if the coefficients are reasonably big to prevent this check triggering for almost straight lines. 

If the sanity check succeeds the area to search for line pixels will be centered around the line found in the last frame (the code can be found in the function `findLinePixelsPreviousFit(imgIn, left_fit, right_fit)`). If it fails we use the sliding window method and try to get a fit uninfluenced by the lines found previously. 

The result can be found here for the project video: [out_project_video.mp4](./out_project_video.mp4)



### Discussion


The algorithm works pretty well on the project_video.mp4 even in more problematic regions e.g. in segments with high curvature, shadow or rough road surface.

However, both challenge videos showed me some shortcomings of my algorithm. Especially the challenge_video.mp4 showed some ambiguities which could be handled better:
1. Gradients and high S channel values should be at the same positions. Currently the binary image is just a pixel-wise OR of the gradient threshold binary image and the S-channel threshold binary image. One could facilitate each to check the other. Object which are breaking the algorithm are e.g. continuous cracks in the surface (those produce edges/gradients) or very bright road limitation (which cause the s-threshold part to fail).
2. In very poorly lit frames (e.g. below bridges) both the gradient and color threshold fails. It might be reasonable to provide a simple confidence variable. This variable could be used to determine if to trust on previous fits or the current one. One simple measure could be the contrast of the current frame.
3. Taking the histogram over the whole y range of the image causes the initial window to not centered correctly in scenes with high-curvature lines. This could be solved easily by applying the histogram only to the lower part of the image.
4. Getting the perspective transform right was a trail and error process. One could facilitate a simple method e.g. the hough transform from the previous project to come up with a initial guess for the ROI/the source points. I might also be interesting to set the ROI based on previous (high-confident) fits.
5. It was in general quite hard to get a reasonably good set of hyper parameters (Thresholds for gradients, transformation source points, size of the sliding windows etc.). I spend considerable time optimizing those with simple criteria based on intuition.  


 