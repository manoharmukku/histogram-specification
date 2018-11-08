# Histogram Specification Implementation

### Usage:
```
$ python hs.py (-i | --image) image_loc (-t | --target) target_name [-h | --help] 
```

- [] => optional
- () => required
- |  => use either of them
- image_loc => location of the input image
- target_name => target distribution name (or) image location for using image as target distribution
- Target distribution names and their descriptions:
    - __uniform__ => Uniform distribution
    - __normal__ => Normal distribution
    - __rayleigh__ => Rayleigh distribution
    - __gamma__ => Gamma distribution
    - __weibull__ => Weibull distribution
    - __beta1__ => Beta dist. with alpha = beta = 0.5
    - __beta2__ => Beta dist. with alpha = 5, beta = 1
    - __beta3__ => Beta dist. with alpha = 8, beta = 2
    - __lognorm__ => Log normal distribution
    - __laplace__ => Laplace distribution

###### Source of images:

- https://unsplash.com/search/photos/grayscale

###### References:

- http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node3.html