# PatchPreprocessor is responsible for randomly sampling
# M x N regions of an image during the training process.
# We apply patch preprocessing when the spatial dimensions of 
# our input images are larger than what the CNN expects - this is a common
# technique to help reduce overfitting, and is, therefore, a form of
# regularization

# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreprocessor:
    def __init__(self, width, height):
        # store the target width and height of the image
        self.width = width
        self.height = height
    
    def preprocess(self, image):
        # extract a random crop from the image with target width
        # and height
        # max_patches = 1 indicates we only need a single random patch from input image
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]
