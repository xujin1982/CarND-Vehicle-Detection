**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform the methods listed below on a labeled training set of images to extract features.
	* a) Color transform,
	* b) Histogram of Oriented Gradients (HOG) feature extraction,
	* c) Binned color features, 
	* d) Histograms of color.
* Train a classifier Linear SVM classifier with the extracted features with normalization and shuffling.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run a pipeline on a video stream.
* Create a heat map of recurring detections frame by frame and filter the detected vehicle based on the previous frames to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/deque_imges.png
[image7]: ./output_images/deque_result.png
[image8]: ./output_images/deque_color_result.jpg
[image9]: ./output_images/ROI.jpg
[video1]: ./output_video/project_video.mp4

---


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `extract_features` function in `Define a function to extract features from a list of images` code cell of the IPython notebook located in `./code/classifier.ipynb`, as well as `get_hog_features` function, `bin_spatial` function and `color_hist` function located in lines 172 through 205 of the file called `./code/utils.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using the `YCrCb` color space and HOG parameters of `orientations=18`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, as well as `spatial_size=(36,36)` for binned color features and  `hist_bins=48` for Histograms of color:

![alt text][image2]

####2. Explain how I settled on my final choice of feature parameters.

I tried various combinations of parameters to create features, including `Color transform`,`orientations`, `pixels_per_cel`, `cells_per_block`, `hog_channel`, `spatial_size` and `hist_bins`.

* `Color transform`: Color space transform. (BGR, HSV, LUV, HLS, YUV and YCrCb)
* `orientations`: Number of orientation bins for HOG. (4-20)
* `pixels_per_cel`: Size (in pixels) of a cell for HOG. ((5,5)-(10,10))
* `cells_per_block`: Number of cells in each block for HOG. ((2,2) - (5,5))
* `hog_channel`: Channel of image used for HOG. (Can be 0, 1, 2, or "ALL")
* `spatial_size`: Spatial binning dimensions for binned color features. ((8,8) - (40,40))
* `hist_bins`: Number of histogram bins for Histograms of color. (16 - 64)

The combination is settled based on the performance of SVC classifier trained in the next step. The final choice of feature parameters are listed below:

* `Color transform`: YCrCb
* `orientations`: 18
* `pixels_per_cel`: 8
* `cells_per_block`: 2
* `hog_channel`: "ALL"
* `spatial_size`: (36, 36)
* `hist_bins`: 48

The total number of features for each image is 14,616. 

####3. Describe how I trained a classifier using my selected HOG features and color features.

The code for this step is contained in the `Train SVC` code cell of the IPython notebook located in `./code/classifier.ipynb`. 

The number of car images in the training data set is 8792 and the number of non-car images is 8968. The training dataset has a balanced positive and negative examples. I shuffle the training dataset and split the training dataset into training and test data with the ratio of 4:1. Based on the training data, I trained a linear SVC using `GridSearchCV` from `sklearn.model_selection`. I tune the parameter `C` of the `LinearSVC` model from `sklearn.svm` with 5 folds cross-validation. Finally, I find the best parameter `C=0.0003`, and get the prediction accuracy on test data equals to 99.21%. 

###Sliding Window Search

####1. Describe how I implemented a sliding window search.  How did I decide what scales to search and how much to overlap windows?

Based on the observation of the video, there are two kinds of vehicle:

* distant car: the vehicles are in far front of the observer. The far front vehicles are small in size and locate in the vertical middle of the image. I utilize sliding windows with small size ((0.75 * 64, 0.75 * 64)=(48, 48)) and high overlap ratio (87.5% overlap, 1 cell to step), as shown in the left part of the following image.  The total number of the windows for distant car is 1648. However, since the distant cars have low resolution in the image, a high overlap ratio has to be implemented to detect them and leads to many false positives. 
* close car: the vehicles are close to the observer.  They are relatively large and easy to be detected. I utilize sliding window with large size ((2 * 64,2 * 64)=(128,128)) and suggested overlap ratio (75% overlap, 2 cell to step), as shown in the right part of the following image. And the total number of the windows for close car is 185.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the final model, two scales ((48,48) and (128, 128)) YCrCb 3-channel HOG features, spatially binned color (36,36) and histograms of color in the feature vector (bins=48) were used to locate a vehicle in an image. Here are an example image:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/project_video.mp4)


####2. Describe how I implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded (threshold=1) that map to identify vehicle positions. The code for this step is contained in the `add_heat` function, `apply_threshold` function, `draw_labeled_bboxes` function and `find_boxes_of_cars` function in `Define functions for finding cars` code cell , as well as the code contained in `Show results after threshold filtering` code cell of the IPython notebook located in `./code/VehicleDetectionTracking.ipynb`. Here is an example of the threshold filtering.

![alt text][image5]


I create a Cars() class to record the detected labels in a deque to filter out the false positives, which are not shown in previous seven frames. This code for this step is contained in the `Define Cars() class for filtering out FP based on previous frames` code cell and `Defome Process_image function` code cell of the IPython notebook located in `./code/VehicleDetectionTracking.ipynb`.


### Here are seven frames and their corresponding binary maps:

![alt text][image6]

### Here is the output of deque result on the integrated binary maps from all seven frames:

![alt text][image7]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image8]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The classifier has a 99.21% accuracy on test data, however, it results a lot of false positives in the video. It is one possible solution for the issue that the classifier needs to be improved to have less false positives.
* The distant cars are difficult to detect. A large number of sliding windows, which introduce a lot of false positives, are implemented to detect the distant cars. It is helpful to detect a distant car by defining a region of interest (ROI) in the center of the image.
* The deque is introduced to filter out the false positives, however, it reduce the sensibility of the vehicle tracking, especially for the new car moving in the sight and for an existing car moving out of the sight. To improve the vehicle tracking, a variable could be defined to record the centroid of each vehicle and track its moving. Also, a ROI to cover the left and right sides of image could be defined to detect the vehicle moving in/out of the sight.
* It reduces the flexibility of sliding windows that computing HOG features for the entire image. It took me a lot of time to tune the parameters of sliding windows to get a good result. It is a trade-off between accuracy and computational complexity. 

###### Here is the suggestion of the ROI:

![alt text][image9]