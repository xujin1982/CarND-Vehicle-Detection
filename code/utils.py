
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from skimage.feature import hog


# In[2]:


class Image_processing:
    def __init__(self, test_img_filenames, chessboard_size, lane_shape):
        """
        Handle camera calibration, distortion correction, perspective warping.
        Handle the proces of creating thresholded binary image with gradients and color transforms. 
        
        :param test_img_filenames: list of file names of the chessboard calibration images.
        :param chessboard_size: the numbers of inseide corners in (x, y)
        :param lane_shape: source points of region of interest (ROI)
        """
        # Get image size
        example_img = cv2.imread(test_img_filenames[0])
        self.img_size = example_img.shape[0:2]
        self.img_height = self.img_size[0]
        self.img_width = self.img_size[1]
        
        
        # Calibration
        self.mtx, self.dist = self.calibration(test_img_filenames,chessboard_size)
        
        # Define bird-eye view transform and its inverse
        top_left, top_right, bottom_left, bottom_right = lane_shape
        self.src = np.float32([top_left, top_right, bottom_right, bottom_left])
        self.dst = np.float32([[self.img_width/4,0], [self.img_width*3/4,0],
                               [self.img_width*3/4,self.img_height-1], [self.img_width/4, self.img_height-1]])
        
        # Get transform matrix and its inverse
        self.bird_view_transform = cv2.getPerspectiveTransform(self.src, self.dst)
        self.inverse_bird_view_transform = cv2.getPerspectiveTransform(self.dst, self.src)
        
    def calibration(self, test_img_filenames, chessboard_size):
        """
        Calibrate the camera using chessboard calibration images.        
        """
        chess_rows, chess_cols = chessboard_size
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chess_rows*chess_cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:chess_cols,0:chess_rows].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Step through the list and search for chessboard corners
        for fname in test_img_filenames:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (chess_cols,chess_rows),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                
        # Camera calibration, given object points, image points, and the shape of the grayscale image
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if not ret:
            raise Exception("Camera calibration unsuccessful.")
        return mtx, dist
    
    def undistort(self,image):
        """
        Undistort camera's raw image
        
        :param image: orignial image
        """
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
    
    def birds_eye(self, undistorted_img):
        """
        Apply a perspective transform to rectify binary image
        
        :param undistorted_img: undistorted image
        """
        return cv2.warpPerspective(undistorted_img, self.bird_view_transform, 
                                   (self.img_width, self.img_height), flags=cv2.INTER_LINEAR)
    
    def inverse_birds_eye(self, bird_view_img):
        """
        Apply a perspective transform to rectify binary image
        
        :param bird_view_img: perspective transformed bird-view image
        """
        return cv2.warpPerspective(bird_view_img, self.inverse_bird_view_transform, 
                                   (self.img_width, self.img_height), flags=cv2.INTER_LINEAR)

    def single_color_detect(self, img, setting):
        result = np.zeros(img.shape[0:2]).astype('uint8')
        for params in setting:
            if params['cspace'] == 'BGR':
                gray = img[:,:,params['channel']]
            elif params['cspace'] == 'GRAY':
                color_t = getattr(cv2, 'COLOR_BGR2{}'.format(params['cspace']))
                gray = cv2.cvtColor(img, color_t)
                if params['channel'] == 'x':
                    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
                    gray = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
                elif params['channel'] == 'y':
                    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
                    gray = np.uint8(255*abs_sobely/np.max(abs_sobely))
                else:
                    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
                    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
                    abs_sobelxy = np.sqrt(abs_sobelx ** 2 + abs_sobely **2)
                    gray = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
            else:
                # Change color space
                color_t = getattr(cv2, 'COLOR_BGR2{}'.format(params['cspace']))
                gray = cv2.cvtColor(img, color_t)[:,:,params['channel']]
            
            # Normalize regions of the image using CLAHE
            clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
            norm_img = clahe.apply(gray)
            
            # Threshold to binary
            ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY)
            result += binary
            
        binary_result = np.zeros(img.shape[0:2]).astype('uint8')
        binary_result[(result > 2)] = 1
        
        return binary_result
        
    def score_pixles(self, img):
        """
        Create thresholded binary image with gradients and color transforms:
            S channel of HLS color space
            B channel of LAB color space
            V channel of HSV color space
            L channel of HLS color space
            
        :param img: bird-view image in BGR.
        """
        white_setting = [
            {'name': 'Sobel_x_w', 'cspace':'GRAY', 'channel': 'x', 'clipLimit': 2.0, 'threshold': 20}, 
            {'name': 'hls_l_w', 'cspace':'HLS', 'channel': 1, 'clipLimit': 8.0, 'threshold': 220},
            {'name': 'hsv_v_w', 'cspace':'HSV', 'channel': 2, 'clipLimit': 4.0, 'threshold': 200},
            {'name': 'bgr_g_w', 'cspace':'BGR', 'channel': 1, 'clipLimit': 8.0, 'threshold': 220}]
        white_result = self.single_color_detect(img, white_setting)
        
        yellow_setting = [
            {'name': 'Sobel_x_y', 'cspace':'GRAY', 'channel': 'x', 'clipLimit': 2.0, 'threshold': 20},
            {'name': 'lab_b_y', 'cspace':'LAB', 'channel': 2, 'clipLimit': 8.0, 'threshold': 180},
            {'name': 'bgr_r_y', 'cspace':'BGR', 'channel': 2, 'clipLimit': 8.0, 'threshold': 220},
            {'name': 'hls_s_y', 'cspace':'HLS', 'channel': 2, 'clipLimit': 4.0, 'threshold': 150}]
        yellow_result = self.single_color_detect(img, yellow_setting)

        binary_result = np.zeros(img.shape[0:2]).astype('uint8')
        binary_result[(white_result == 1) | (yellow_result == 1)] = 1
        return binary_result


# In[3]:


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

