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
import matplotlib.pyplot as plt

def help():
    print ("-------------------------------------------------------------")
    print ("Usage: $ python hs.py -i image_loc -t target_dist_name_or_loc")
    print ("-------------------------------------------------------------")
    print ("Following are the target distributions names to use:")
    print ("----------------------------------------------------")
    print ("uniform => Uniform distribution")
    print ("normal => Normal distribution")
    print ("rayleigh => Rayleigh distribution")
    print ("gamma => Gamma distribution")
    print ("weibull => Weibull distribution")
    print ("beta1 => Beta(a=b=0.5) distribution")
    print ("beta2 => Beta(a=5,b=1) distribution")
    print ("lognorm => Lognorm distribution")
    print ("laplace => Laplace distribution")
    print ("beta3 => Beta(a=8,b=2) distribution")
    print ("target_image_loc => Target image location")

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
        # Import the package of the target distribution
        from scipy.stats import uniform

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
        # Import the package of the target distribution
        from scipy.stats import norm

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
        # Import the package of the target distribution
        from scipy.stats import rayleigh

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
        # Import the package of the target distribution
        from scipy.stats import gamma

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
        # Import the package of the target distribution
        from scipy.stats import weibull_min

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
        # Import the package of the target distribution
        from scipy.stats import beta

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
        # Import the package of the target distribution
        from scipy.stats import beta

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
        # Import the package of the target distribution
        from scipy.stats import lognorm

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
        # Import the package of the target distribution
        from scipy.stats import laplace

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
        # Import the package of the target distribution
        from scipy.stats import beta

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

    # Create the transformed image using the img's pixel values and the lookup table
    trans_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            trans_img[i][j] = lookup[img[i][j]]

    # Write the transformed image to a png file
    cv2.imwrite('images/transformed.png', trans_img)

    # Plot the input image and the target image in one plot
    input_img_resized = cv2.resize(img, (0,0), None, 0.25, 0.25)
    trans_img_resized = cv2.resize(trans_img, (0,0), None, 0.25, 0.25)
    numpy_horiz = np.hstack((input_img_resized, trans_img_resized))
    cv2.imshow('Input image ------------------------ Trans image', numpy_horiz)
    cv2.waitKey(25)

    # Calculate the transformed image's histogram
    trans_hist = cv2.calcHist([trans_img], [0], None, [256], [0,256])

    # Normalize the transformed image's histogram
    total = sum(trans_hist)
    trans_hist /= total

    # Convert cum_input_hist to matrix for plotting
    cum_input_hist_matrix = np.ndarray(shape=(256,1))
    for i in range(0,256):
        cum_input_hist_matrix[i][0] = cum_input_hist[i]

    # Calculate the cum transformed histogram for plotting
    cum_trans_hist = np.ndarray(shape=(256,1))
    cum = 0.0
    for i in range(0,256):
        cum += trans_hist[i][0]
        cum_trans_hist[i][0] = cum

    # Convert cum_target_hist to matrix for plotting
    cum_target_hist_matrix = np.ndarray(shape=(256,1))
    for i in range(0,256):
        cum_target_hist_matrix[i][0] = cum_target_hist[i]

    plt.subplot(2, 3, 1)
    plt.title('Original hist')
    plt.plot(input_hist)

    plt.subplot(2, 3, 2)
    plt.title('Original cdf')
    plt.plot(cum_input_hist_matrix)

    plt.subplot(2, 3, 3)
    plt.title('Target pdf')
    plt.plot(target_hist)

    plt.subplot(2, 3, 4)
    plt.title('Transformed hist')
    plt.plot(trans_hist)

    plt.subplot(2, 3, 5)
    plt.title('Transformed cdf')
    plt.plot(cum_trans_hist)

    plt.subplot(2, 3, 6)
    plt.title('Target cdf')
    plt.plot(cum_target_hist_matrix)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])