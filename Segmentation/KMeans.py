class KmeansSegmentation:

    def segmentation_grey(self, image, k = 2):
        """Performs segmentation of an grey level input image using KMeans algorithm, using the intensity of the pixels as features
        takes as input:
        image: a grey scale image
        return an segemented image
        -----------------------------------------------------
        Sample implementation for K-means
        1. Initialize cluster centers
        2. Assign pixels to cluster based on (intensity) proximity to cluster centers
        3. While new cluster centers have moved:
            1. compute new cluster centers based on the pixels
            2. Assign pixels to cluster based on the proximity to new cluster centers

        """

        return image


    def segmentation_rgb(self, image, k=2):
        """Performs segmentation of a color input image using KMeans algorithm, using the intensity of the pixels (R, G, B)
        as features
        takes as input:
        image: a color image
        return an segemented image"""

        return image
