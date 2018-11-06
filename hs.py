'''
Author: Manohar Mukku
Date: 06.11.2018
Desc: Histogram Specification
GitHub: https://github.com/manoharmukku/histogram-specification
'''

import sys
import getopt
import numpy as np
import cv2

def get_arguments(argv):
    # Get the command line arguments
    try:
        opts, args = getopt.getopt(argv, "hi:t:", ["help", "image=", "target="])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    image = None
    target = None

    # Parse the command line arguments
    for opt, arg in opts:
        if (opt in ["-h", "--help"]):
            help()
            sys.exit()
        elif (opt in ["-i", "--image"]):
            image = arg
        elif (opt in ["-t", "--target"]):
            target = arg
        else:
            print ("Invalid syntax")
            help()
            sys.exit()

    return image, target

def main(argv):
    # Get and parse the command line arguments
    image_loc, target_hist = get_arguments(argv)

    # Read the image
    img = cv2.imread(image_loc, 0)

    # Calculate the input images' histogram
    input_hist = cv2.calcHist([img], [0], None, [256], [0,256])

    # Normalize the input histogram
    total = sum(input_hist)
    input_hist /= total

    # Calculate the cumulative input histogram
    cum_input_hist = []
    cum = 0.0
    for i in range(len(input_hist)):
        cum += input_hist[i][0]
        cum_input_hist.append(cum)

if __name__ == "__main__":
    main(sys.argv[1:])