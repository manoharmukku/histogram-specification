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
from scipy.stats import uniform

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
    image_loc, target_name = get_arguments(argv)

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

    target_hist = []
    cum_target_hist = []

    # Calculate the target histogram
    if (target_name == "uniform"):
        # Create uniform distribution object
        unif_dist = uniform(0, 246)

        # Calculate the target histogram
        for i in range(0, 246):
            x = unif_dist.pdf(i)
            target_hist.append(x)
        for i in range(246, 256):
            target_hist.append(0)

        # Calculate the cumulative target histogram
        cum = 0.0
        for i in range(len(target_hist)):
            cum += target_hist[i]
            cum_target_hist.append(cum)


    # Obtain the mapping from the input hist to target hist
    lookup = {}
    for i in range(len(cum_input_hist)):
        min_val = abs(cum_target_hist[0] - cum_input_hist[i])
        min_j = 0

        for j in range(1, len(cum_target_hist)):
            val = abs(cum_target_hist[j] - cum_input_hist[i])
            if (val < min_val):
                min_val = val
                min_j = j

        lookup[i] = min_j

    # Update the img's pixel values to target specification using the lookup table
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = lookup[img[i][j]]

    # Write the target image to a png file
    cv2.imwrite('images/target.png', img)

if __name__ == "__main__":
    main(sys.argv[1:])