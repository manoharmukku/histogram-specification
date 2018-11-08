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
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import gamma
from scipy.stats import weibull_min
from scipy.stats import beta
from scipy.stats import lognorm
from scipy.stats import laplace
from matplotlib import pyplot as plt

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

    # Read the input image
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

    # Calculate the variance of the image
    input_img_var = np.var(img)

    # Calculate the variance of the image square
    input_img_sqr_var = np.var(img**2)

    # Calculate the target dist for diff dist's
    target_dist = []
    target_hist = None
    if (target_name == "uniform"):
        # Create uniform distribution object
        unif_dist = uniform(0, 246)

        # Calculate the target distribution
        for i in range(0, 246):
            x = unif_dist.pdf(i)
            target_dist.append(x)
        for i in range(246, 256):
            target_dist.append(0)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

    elif (target_name == "normal"):
        # Create standard normal distribution object
        norm_dist = norm(0, 1)

        # Calculate the target distribution
        for i in range(0, 256):
            x = norm_dist.pdf(i/42.0 - 3)
            target_dist.append(x)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "rayleigh"):
        # Create rayleigh distribution object
        rayleigh_dist = rayleigh(0.5)

        # Calculate the target distribution
        for i in range(0, 256):
            x = rayleigh_dist.pdf(i/128.0)
            target_dist.append(x)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "gamma"):
        # Create gamma distribution object
        gamma_dist = gamma(0.5, 0, 1.0)

        # Calculate the target distribution
        target_dist.append(1)
        for i in range(1, 256):
            x = gamma_dist.pdf(i/256.0)
            target_dist.append(x)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "weibull"):
        # Create weibull distribution object
        weibull_dist = weibull_min(c=1.4, scale=input_img_var)

        # Calculate the target distribution
        for i in range(0, 256):
            x = weibull_dist.pdf(i/256.0)
            target_dist.append(x)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "beta1"):
        # Create beta distribution object
        beta_dist = beta(0.5, 0.5)

        # Calculate the target distribution
        target_dist.append(6)
        for i in range(1, 255):
            x = beta_dist.pdf(i/256.0)
            target_dist.append(x)
        target_dist.append(6)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "beta2"):
        # Create beta distribution object
        beta_dist = beta(5, 1)

        # Calculate the target distribution
        for i in range(0, 255):
            x = beta_dist.pdf(i/256.0)
            target_dist.append(x)
        target_dist.append(6)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "lognorm"):
        # Create lognorm distribution object
        lognorm_dist = lognorm(1)

        # Calculate the target distribution
        for i in range(0, 256):
            x = lognorm_dist.pdf(i/100.0)
            target_dist.append(x)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "laplace"):
        # Create lognorm distribution object
        laplace_dist = laplace(4)

        # Calculate the target distribution
        target_dist.append(0)
        for i in range(1, 256):
            x = laplace_dist.pdf(i/256.0)
            target_dist.append(x)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    elif (target_name == "beta3"):
        # Create beta distribution object
        beta_dist = beta(8, 2)

        # Calculate the target distribution
        for i in range(0, 255):
            x = beta_dist.pdf(i/256.0)
            target_dist.append(x)
        target_dist.append(0)

        # Calculate the target histogram
        target_hist = np.ndarray(shape=(256,1))
        for i in range(0,256):
            target_hist[i][0] = target_dist[i]

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total

    else: # Image is a target distribution
        # Read the image
        target_dist = cv2.imread(target_name, 0)

        # Create target histogram from the image
        target_hist = cv2.calcHist([target_dist], [0], None, [256], [0,256])

        # Normalize the target histogram
        total = sum(target_hist)
        target_hist /= total
        

    # Calculate the cumulative target histogram
    cum_target_hist = []
    cum = 0.0
    for i in range(len(target_hist)):
        cum += target_hist[i][0]
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

    # Create the target image using the img's pixel values and the lookup table
    spec_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            spec_img[i][j] = lookup[img[i][j]]

    # Write the target image to a png file
    cv2.imwrite('images/target.png', spec_img)

    # Plot the input image and the target image in one plot
    input_img = cv2.resize(img, (0,0), None, 0.25, 0.25)
    target_img = cv2.resize(spec_img, (0,0), None, 0.25, 0.25)
    numpy_horiz = np.hstack((input_img, target_img))
    cv2.imshow('Input image ------------------------ Target image', numpy_horiz)
    cv2.waitKey()

    # Calculate the specificated images' histogram
    spec_hist = cv2.calcHist([spec_img], [0], None, [256], [0,256])

    # Normalize the input histogram
    total = sum(spec_hist)
    spec_hist /= total

    # Calculate the cum input histogram
    cum_input_hist2 = np.ndarray(shape=(256,1))
    for i in range(0,256):
        cum_input_hist2[i][0] = cum_input_hist[i]

    # Calculate the cum target histogram
    cum_target_hist2 = np.ndarray(shape=(256,1))
    cum = 0.0
    for i in range(0,256):
        cum += spec_hist[i][0]
        cum_target_hist2[i][0] = cum

    cum_target_hist3 = np.ndarray(shape=(256,1))
    cum = 0.0
    for i in range(0,256):
        cum_target_hist3[i][0] = cum_target_hist[i]

    plt.subplot(2, 3, 1)
    plt.title('Original hist')
    plt.plot(input_hist)

    plt.subplot(2, 3, 2)
    plt.title('Original cdf')
    plt.plot(cum_input_hist2)

    plt.subplot(2, 3, 3)
    plt.title('Target pdf')
    plt.plot(target_hist)

    plt.subplot(2, 3, 4)
    plt.title('Transformed hist')
    plt.plot(spec_hist)

    plt.subplot(2, 3, 5)
    plt.title('Transformed cdf')
    plt.plot(cum_target_hist2)

    plt.subplot(2, 3, 6)
    plt.title('Target cdf')
    plt.plot(cum_target_hist3)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])