import skimage.color
import skimage.io as io
import numpy as np
from scipy import interpolate, ndimage, signal
import skimage.metrics

####################################################################################
# Part 1: Implement demosaicing using linear interpolation
# load image
raw_image = io.imread('lighthouse_RAW_noisy_sigma0.01.png').astype(np.float64)/255   # float 0 to 1
n_row, n_col = raw_image.shape
x_axis = np.arange(n_col)
y_axis = np.arange(n_row)

# interpolate and construct the red channel
x_red = np.arange(0, n_col, 2)
y_red = np.arange(0, n_row, 2)
red_pixels = raw_image[::2, ::2]
f_red = interpolate.interp2d(x_red, y_red, red_pixels)
red = f_red(x_axis, y_axis)

# interpolate and construct the blue channel
x_blue = np.arange(1, n_col, 2)
y_blue = np.arange(1, n_row, 2)
blue_pixels = raw_image[1::2, 1::2]
f_blue = interpolate.interp2d(x_blue, y_blue, blue_pixels)
blue = f_blue(x_axis, y_axis)

# interpolate and construct the green channel
x_green, y_green = np.meshgrid(x_axis, y_axis)
green = np.where((x_green+y_green) % 2 == 1, 1, 0)
green = np.multiply(green, raw_image)
green_up = np.roll(green, -1, axis=0); green_up[-1, :] = np.zeros(n_col)
green_down = np.roll(green, 1, axis=0); green_down[0, :] = np.zeros(n_col)
green_left = np.roll(green, -1, axis=1); green_left[:, -1] = np.zeros(n_row)
green_right = np.roll(green, 1, axis=1); green_right[:, 0] = np.zeros(n_row)
scalar = np.ones_like(green)*4  # what scalar to divide for each pixel
scalar[0, :], scalar[:, 0], scalar[-1, :], scalar[:, -1] = 3, 3, 3, 3
scalar[0, 0], scalar[0, -1], scalar[-1, 0], scalar[-1, -1] = 2, 2, 2, 2
green = np.divide(green_up+green_down+green_left+green_right, scalar) + green

# combine the 3 channels into an RGB image
linear_interp = np.stack([red, green, blue], axis=2)  # float 0 to 1
# apply a gamma correction to convert it to a proper sRGB image
gamma_interp = np.power(linear_interp, 1/2.2)  # float 0 to 1
io.imsave("task2_part1.png", np.clip(255*gamma_interp, a_min=0, a_max=255.))
# calculate the PSNR
original_img = io.imread('lighthouse.png').astype(np.float64)/255  # float 0 to 1
print(skimage.metrics.peak_signal_noise_ratio(original_img, gamma_interp))

####################################################################################
# Part 2:
# convert RGB to YCrCb, luminance is given by Y and chrominance by Cb and Cr
# Python rgb2ycbcr and ycbcr2rgb use float from 0 to 1
ycrcb_linear = skimage.color.rgb2ycbcr(linear_interp)
# low-pass or median filter the chrominance channels
ycrcb_linear[:, :, 1] = ndimage.median_filter(ycrcb_linear[:, :, 1], size=12)
ycrcb_linear[:, :, 2] = ndimage.median_filter(ycrcb_linear[:, :, 2], size=12)
# convert back to RGB and apply gamma correction
smooth_interp = skimage.color.ycbcr2rgb(ycrcb_linear)  # float from 0 to 1
smooth_interp = np.clip(smooth_interp, 0., 1.)
#smooth_interp = smooth_interp.astype(np.float64)/255
smooth_interp = np.power(smooth_interp, 1/2.2)
io.imsave("task2_part2.png", np.clip(255*smooth_interp, a_min=0, a_max=255.))
# calculate the PSNR
print(skimage.metrics.peak_signal_noise_ratio(original_img, smooth_interp))

####################################################################################
# Part 3:
# construct the filter/convolve kernels
GR_GB_kernel = (1/8) * np.array([[0., 0., -1., 0., 0.],
                                 [0., 0., 2., 0., 0.],
                                 [-1., 2., 4., 2., -1.],
                                 [0., 0., 2., 0., 0.],
                                 [0., 0., -1., 0., 0.]])
RgRB_BgBR_kernel = (1/8) * np.array([[0., 0., .5, 0., 0.],
                                     [0., -1., 0., -1., 0.],
                                     [-1., 4., 5., 4., -1.],
                                     [0., -1., 0., -1., 0.],
                                     [0., 0., .5, 0., 0.]])
RgBR_BgRB_kernel = np.transpose(RgRB_BgBR_kernel)
RbBB_BrRR_kernel = (1/8) * np.array([[0., 0., -1.5, 0., 0.],
                                     [0., 2., 0., 2., 0.],
                                     [-1.5, 0., 6., 0., -1.5],
                                     [0., 2., 0., 2., 0.],
                                     [0., 0., -1.5, 0., 0.]])
# get all red, green, blue pixel respectively
x_grid, y_grid = np.meshgrid(x_axis, y_axis)
r_filter = np.where(np.logical_and(x_grid % 2 == 0, y_grid % 2 == 0), 1, 0)
r_raw = np.multiply(r_filter, raw_image)
g_filter = np.where((x_grid+y_grid) % 2 == 1, 1, 0)
g_raw = np.multiply(g_filter, raw_image)
b_filter = np.where(np.logical_and(x_grid % 2 == 1, y_grid % 2 == 1), 1, 0)
b_raw = np.multiply(b_filter, raw_image)
# build the green channel
G = np.where(np.logical_or(r_filter == 1, b_filter == 1),
             signal.convolve2d(raw_image, GR_GB_kernel, mode='same'), g_raw)
# build the red channel
Rrow_Bcol_filter = np.where(np.logical_and(x_grid % 2 == 1, y_grid % 2 == 0), 1, 0)
Brow_Rcol_filter = np.where(np.logical_and(x_grid % 2 == 0, y_grid % 2 == 1), 1, 0)
Brow_Bcol_filter = np.where(np.logical_and(x_grid % 2 == 1, y_grid % 2 == 1), 1, 0)
R = np.where(Rrow_Bcol_filter == 1, signal.convolve2d(raw_image, RgRB_BgBR_kernel, mode='same'), r_raw)
R = np.where(Brow_Rcol_filter == 1, signal.convolve2d(raw_image, RgBR_BgRB_kernel, mode='same'), R)
R = np.where(Brow_Bcol_filter == 1, signal.convolve2d(raw_image, RbBB_BrRR_kernel, mode='same'), R)
# build the blue channel
Rrow_Rcol_filter = np.where(np.logical_and(x_grid % 2 == 0, y_grid % 2 == 0), 1, 0)
B = np.where(Brow_Rcol_filter == 1, signal.convolve2d(raw_image, RgRB_BgBR_kernel, mode='same'), b_raw)
B = np.where(Rrow_Bcol_filter == 1, signal.convolve2d(raw_image, RgBR_BgRB_kernel, mode='same'), B)
B = np.where(Rrow_Rcol_filter == 1, signal.convolve2d(raw_image, RbBB_BrRR_kernel, mode='same'), B)
# combine the 3 channels into an RGB image
high_linear_interp = np.stack([R, G, B], axis=2)
# apply a gamma correction to convert it to a proper sRGB image
high_linear_interp = np.clip(high_linear_interp, 0., 1.)
high_gamma_interp = np.power(high_linear_interp, 1/2.2)
io.imsave("task2_part3.png", np.clip(255*high_gamma_interp, a_min=0, a_max=255.))
# calculate the PSNR
print(skimage.metrics.peak_signal_noise_ratio(original_img, high_gamma_interp))
