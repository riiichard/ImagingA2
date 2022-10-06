import numpy as np
from fspecial import fspecial_gaussian_2d


def inbounds(img, y, x):
    return 0 <= y and y < img.shape[0] and \
           0 <= x and x < img.shape[1]


def comparePatches(patch1, patch2, kernel, sigma):
    return np.exp(-np.sum(kernel*(patch1 - patch2) ** 2)/(2*sigma**2))


def nonlocalmeans(img, searchWindowRadius, averageFilterRadius, sigma, nlmSigma):
    # Initialize output to 0
    out = np.zeros_like(img)
    # Pad image to reduce boundary artifacts
    pad = max(averageFilterRadius, searchWindowRadius)
    imgPad = np.pad(img, pad)
    imgPad = imgPad[..., pad:-pad] # Don't pad third channel

    # Smoothing kernel
    filtSize = (2*averageFilterRadius + 1, 2*averageFilterRadius + 1)
    kernel = fspecial_gaussian_2d(filtSize, sigma)
    # Add third axis for broadcasting
    kernel = kernel[:, :, np.newaxis]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerPatch = imgPad[y+pad-averageFilterRadius:y+pad+averageFilterRadius+1,
                                 x+pad-averageFilterRadius:x+pad+averageFilterRadius+1,
                                 :]
            # Go over a window around the current pixel, compute weights
            # based on difference of patches, sum the weighted intensity
            # Hint: Do NOT include the patches centered at the current pixel
            # in this loop, it will throw off the weights
            weights = np.zeros((2*searchWindowRadius+1, 2*searchWindowRadius+1, 1))

            # # This makes it a bit better: Add current pixel as well with max weight
            # # computed from all other neighborhoods.
            # max_weight = 0

            # out[y, x, :] = 0. # TODO: Replace with your code.
            # construct the window want to find all similarities
            window_width = 2 * searchWindowRadius + 1
            window_y_min, window_y_max = y+pad-searchWindowRadius, y+pad+searchWindowRadius+1
            window_x_min, window_x_max = x+pad-searchWindowRadius, x+pad+searchWindowRadius+1
            window = imgPad[window_y_min:window_y_max, window_x_min:window_x_max, :]

            # for each pixel in the window
            for i in np.arange(window_y_min, window_y_max):
                for j in np.arange(window_x_min, window_x_max):
                    # skip if currently at the center pixel
                    if not (i == y+searchWindowRadius and j == x+searchWindowRadius):

                        # if current pixel is not one in original image, weight is 0
                        if not inbounds(img, i-pad, j-pad):
                            weights[i-window_y_min, j-window_x_min] = 0
                            continue

                        # compare the patch with center patch and compute the weight
                        patch_compare = imgPad[i-averageFilterRadius: i+averageFilterRadius+1,
                                               j-averageFilterRadius: j+averageFilterRadius+1, 0]
                        pixel_weight = comparePatches(centerPatch[:, :, 0], patch_compare, kernel[:, :, 0], nlmSigma)
                        weights[i-window_y_min, j-window_x_min] = pixel_weight

            # This makes it a bit better: Add current pixel as well with max weight
            # computed from all other neighborhoods.
            weights[window_width // 2, window_width // 2] = np.amax(weights)
            # now have the weights for all pixels in the window
            out[y, x] = np.sum(np.multiply(window, weights)) / np.sum(weights)

    return out
