# USAGE
# python compare.py
'''
this is uses some image comparison algoriths to compare
images and uses a simple black/while pixel function
to find the mmse to use to train knn classifyer
'''


# import the necessary packages
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, PIL, time, math, operator
from functools import reduce
from PIL import Image
from shutil import copy
import pickle

from pdf2png import pdf2img

def whiteblack(image):
    img = image
    height, width = img.shape
    data = img
    pixelList = []
    pix = img
    for x in range(height):
        for y in range(width):
            pixelList.append((x, y, pix[x,y]))            
    black_coords = []
    x_coords = []
    y_coords = []
    blck_count = 0
    for i in pixelList:
        if i[2] < 130:
            blck_count += 1
            black_coords.append((i[0],i[1]))
            x_coords.append(i[0])
            y_coords.append(i[1])
    pixels = len(pixelList)
    blck_cover = blck_count/len(pixelList)
    return blck_cover


def perc_diff(imageA, imageB):
    img1 = Image.open(imageA)
    img2 = Image.open(imageB)
    h1 = img1.histogram()
    h2 = img2.histogram()
     
    rms = math.sqrt(reduce(operator.add,
    map(lambda a,b: (a-b)**2, h1, h2))/len(h1))
    return rms

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	

	# setup the figure
##	fig = plt.figure(title)
##	plt.suptitle("MSE: %.2f, SSIM: %.2f, Percent Diff: %.2f" % (m, s , p))
##
##	# show first image
##	ax = fig.add_subplot(1, 2, 1)
##	ax.set_title('Black pixel cover: %.4f' % blck1)
##	plt.imshow(imageA, cmap = plt.cm.gray)
##	plt.axis("off")
##
##	# show the second image
##	ax = fig.add_subplot(1, 2, 2)
##	ax.set_title('Black pixel cover: %.4f' % blck2)
##	plt.imshow(imageB, cmap = plt.cm.gray)
##	plt.axis("off")
##	print("MSE:", m)
##	print("SSIM:", s)
##	print("Diff:", p)
##	print("black:", blck2)
##	print("")
##        
	# show the images
	#plt.show()
	return m,s

def resize(imagepath, shape):
    image = Image.open(imagepath)                   
    imResize = image.resize(shape, Image.ANTIALIAS)
    open_cv_image = np.array(imResize)
    #open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image
    
    
# load the images -- the original, the original + contrast,
def find_mmse(mmse_img,false, path):
    mmse_list = []
    non_mmse_list = []
    for root,dirs,files in os.walk(path):
        if 'FLAGGED' in path:
            continue
        #dirs[:] = [d for d in dirs if d not in dir_list]
        #dir_list.append(dirs)
        for folder in dirs:
            #print(folder)
            for root,dirs,files in os.walk(os.path.join(path,folder)):
                png_list = files
                for i in png_list:
                    #print(i)
                    file1 = mmse_img
                    file2 = os.path.join(path, folder, i)

                    img1 = resize(file1, (32,32))
                    img2 = resize(file2, (32,32))
                    false_pos = resize(false, (32,32))

    ##                img1 = cv2.imread(file1)
    ##                img2 = cv2.imread(file2)
                    
                    #p = perc_diff(file1, file2)
                    blck1 = whiteblack(img1)
                    blck2 = whiteblack(img2)
                    
##                    original = cv2.imread(file1)
##                    test_image = cv2.imread(file2)
##
##                    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
##                    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

                    # initialize the figure
                    #fig = plt.figure("Images")
##                    images = ("Original", original), ("Test Image", test_image)

                    # show the figure)
                    #compare_images(original, original, "Original vs. Original")
                    fm, fs = compare_images(false_pos, img2, "False positive")
                    #print(fs)
                    m,s = compare_images(img1, img2, "Original vs. Test Image")
                    if blck2 < 0.03 and s > 0.6 and fs < .80:
                        mmse_list.append(file2)
                    else:
                        non_mmse_list.append(file2)
    return mmse_list, non_mmse_list

def save_training(file_list, path, folder):
    save_folder = os.path.join(path, folder)
    if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    for file in file_list:
        proj_file = os.path.dirname(file)
        img_file = os.path.basename(file)
        proj_file = proj_file.split('\\')[-1]
        save_name = os.path.join(save_folder, proj_file + img_file)
        copy(file, save_name)


start_t = time.clock()

#pdf2img(r'C:\Users\KinectProcessing\Desktop\test_mmse_pdf')
path = r'C:\Users\KinectProcessing\Desktop\test_mmse_pdf\extracted_images'
file1 = r'C:\Users\KinectProcessing\Desktop\test_mmse_pdf\extracted_images\435964305  FU2  09-19-07\img-30.png'
false = r'C:\Users\KinectProcessing\Desktop\test_mmse_pdf\extracted_images\737634767  F-U 3  03-01-06\img-10.png'

mmse_list, non_mmse_list = find_mmse(file1,false, path)
##
##pickle_out = open('mmse.pickle', 'wb')
##pickle.dump(mmse_list, pickle_out)
##pickle_out.close()

end_t = time.clock()
print("")
print("TOTAL TIME=", round(end_t - start_t, 2))
save_training(mmse_list, path, 'mmse_training')
save_training(non_mmse_list, path, 'non_mmse')

##pickle_in = open('mmse.pickle', 'rb')
##mmse_list = pickle.load(pickle_in)

##    org_img = cv2.imread(file1)
##    found_img = cv2.imread(file)
##    
##    # setup the figure
##    fig = plt.figure()
##    plt.suptitle("MMSE MATCH")
##
##    # show first image
##    ax = fig.add_subplot(1, 2, 1)
##    ax.set_title('original')
##    plt.imshow(org_img, cmap = plt.cm.gray)
##    plt.axis("off")
##
##    # show the second image
##    ax = fig.add_subplot(1, 2, 2)
##    ax.set_title('test_image')
##    plt.imshow(found_img, cmap = plt.cm.gray)
##    plt.axis("off")
##    
##    plt.show()


