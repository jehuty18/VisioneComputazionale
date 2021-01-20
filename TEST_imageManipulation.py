import os
import numpy as np
import image_slicer
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage import img_as_float
from skimage import color
from skimage.morphology import reconstruction
from skimage.io import imread
from itertools import combinations
import matplotlib.pyplot as plt
from sewar.full_ref import uqi, mse

image_path = "./src/resources/sample.png"
#image_path = "./src/resources/IM_1.png"
#image_path = "./src/resources/IM_1t.png"
#image_path = "./src/resources/tremiti.jpg"
N = 12 # number of slices
dir = "./test/resources"

def read_image(image_path):
    image = imread(image_path)
    return image

def gaussian_filter1(image):
    image = img_as_float(image)
    image = gaussian_filter(image,100)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')
    return dilated

def filtered_image(image):
    image1 = image
    image2 = gaussian_filter1(image)
    image3 =  image1 - image2

    print("START filtered_image")
    print("image1")
    #print(image1)
    plt.figure()
    plt.imshow(image1)
    im1 = np.array(image1)
    print(im1.flatten())
    print("image2")
    print("----------------------")
    print(mse(image1,image2))
    print("----------------------")
    #print(image2)
    plt.figure()
    plt.imshow(image2)
    print("img3")
    #print(img3)
    plt.figure()
    plt.imshow(image3)
    print("END filtered_image")
    #io.imsave("out.png", image3)
    io.imsave("out.png", (color.convert_colorspace(image3, 'HSV', 'RGB')*255).astype(np.uint8))
    return "out.png"

print('sliced_image creation...')
sliced_images = image_slicer.slice(filtered_image(read_image(image_path)),N, save=False)
print('sliced_image created')

print('sliced_image saving...')
image_slicer.save_tiles(sliced_images, directory=dir, prefix='slice')
print('sliced_image saved')

list_files = []
for file in os.listdir(dir):
    if file.lower().endswith('.png'):
        list_files.append(file)
for i in combinations(list_files,2):
    img1 = read_image(dir + '/' + i[0])
    img2 = read_image(dir + '/' + i[1])
    diff = img1 - img2

    diff_btwn_img_data = np.linalg.norm(diff,axis=1)
    #print("diff between " + str(i) + " two images is " + str(np.mean(diff_btwn_img_data)))

plt.show()