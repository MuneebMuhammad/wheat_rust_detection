import skimage.segmentation
from matplotlib import pyplot as plt


img = plt.imread('images/1.jpeg')
res2 = skimage.segmentation.felzenszwalb(img, scale=50)
res3 = skimage.segmentation.felzenszwalb(img, scale=1000)
plt.imshow(res2)
plt.show()
plt.imshow(res3)
plt.show()
