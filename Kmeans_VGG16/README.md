### Fully Unsupervised Learning method to detect rust disease in wheat crop

### Method

- First RGB image is converted to patches and the patches are given to VGG16 model
- VGG16 model generates feature vector for the patch
- The feature vector is provided to K-means for clustering
- K-means clustering provides psudo labels for VGG16
- Only those psudo labels are used whose feature vectors are closer to the centroids

### Dataset

[**Link**](https://drive.google.com/file/d/1a0-uZvADu6q3S6FUCgb8I-AagqVK4Dj7/view?usp=sharing) for testing data <br>
Give the path of wheat crop images in DATSET_LOCATION

### Files

Run **train.py** for training VGG16 and K-means model<br>
Run **test.py** for testing the model
