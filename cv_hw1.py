"""cv_hw1.py: Starter file to run howework 2"""

#Example Usage: ./cv_hw1 -i image -k clusters -m grey
#Example Usage: python cv_hw1 -i image -k clusters -m rgb


__author__      = "Pranav Mantini"
__email__ = "pmantini@uh.edu"
__version__ = "1.0.0"

import cv2
import sys
from Segmentation.KMeans import KmeansSegmentation
from datetime import datetime

def display_image(window_name, image):
    """A function to display image
    Note: only for debuggging do not use for submission, Circle CI will fail"""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     kmeans method and writes the output image"""

    #Parse input arguments
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-i", "--image", dest="image",
                        help="specify the name of the image", metavar="IMAGE")
    parser.add_argument("-k", "--clusters", dest="clusters",
                        help="Specify the number of clusters (k)", metavar="CLUSTERS")
    parser.add_argument("-m", "--model", dest="model",
                        help="Specify the model rgb, grey", metavar="COLOR")

    args = parser.parse_args()

    #Load image
    if args.image is None:
        print("Please specify the name of image")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        image_name = args.image.split(".")[0]
        input_image = cv2.imread(args.image)


    if args.clusters is None:
        print("Number of clusters not specified using 2")
        print("use the -h option to see usage information")
        clusters = 2
    else:
        clusters = int(args.clusters)

    # Check resize scale parametes
    if args.model is None:
        print("Model not specified using default (grey)")
        print("use the -h option to see usage information")
        model = 'grey'
    elif args.model not in ['rgb', 'grey']:
        print("Unknown color model, using default (grey)")
        print("use the -h option to see usage information")
        model = 'grey'
    else:
        model = args.model

    Segementation_object = KmeansSegmentation()

    output = None
    if model == 'grey':
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        output = Segementation_object.segmentation_grey(input_image, clusters)
    else:
        output = Segementation_object.segmentation_rgb(input_image, clusters)


    #Write output file
    output_dir = 'output/'
    output_image_name = output_dir+image_name+"_"+model+"_"+str(clusters)+"_"+datetime.now().strftime("%m%d-%H%M%S")+".jpg"
    cv2.imwrite(output_image_name, output)


if __name__ == "__main__":
    main()
