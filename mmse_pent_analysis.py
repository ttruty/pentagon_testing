from PIL import Image
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import pent_corner_detection as corner
import hough_trans_line as hough

from skimage import io
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
from skimage import data
from skimage.color import rgb2gray
import skimage.filters as filters

''' this is a  demo of all of the methos used to finding the corners in the pentagons'''

path = r'C:\Users\KinectProcessing\Desktop\mmse_pent_analysis\pent.jpg'
img = Image.open(path)
pents = io.imread(path)

## pixel data info
data = img.getdata()
height, width = img.size

blank_img = Image.new('RGBA', (width, height))

pixelList = []
pix = img.load()

for x in range(height):
    for y in range(width):
        pixelList.append((x, y, pix[x,y]))
        
black_coords = []
x_coords = []
y_coords = []

for i in pixelList:
    if i[2][0] < 50:
        black_coords.append((i[0],i[1]))
        x_coords.append(i[0])
        y_coords.append(i[1])

#plt.scatter(x_coords,y_coords)


'''cv2 find corners method with GOOD '''
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#cv2.imshow("bw_Image", im_bw)
corners = corner.find_corners(im_bw)

'''hough probablity transform'''
edges, lines = hough.h_line(pents)

'''manually segmenting'''
def segment_bound(left_bound, right_bound, low_bound, high_bound):
    segX = []
    segY = []
    for i in black_coords:
        if i[0] > left_bound and i[0] < right_bound and i[1] > low_bound and i[1] < high_bound:
            segX.append(i[0])
            segY.append(i[1])
    return segX, segY


## this is quick and dirty can clean up with functions if need be but it works to get the xy of intersetion
# segment 1
def seg_intersect():
    seg1X, seg1Y = segment_bound(147.4, 285.7, 305, 345)
    startline = min(seg1X)
    endline = max(seg1X)
    p_coeff = np.polyfit(seg1X, seg1Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 20, endline + 20)
    seg1m, seg1b = list(p)
    #seg1, = plt.plot( x, p(x), label='Segment1', color='g' )
    seg1_line = ( x, p(x))
    
    # segment 2
    seg2X, seg2Y = segment_bound(131, 140, 168, 300)
    startline = min(seg2X)
    endline = max(seg2X)
    p_coeff = np.polyfit(seg2X, seg2Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 5, endline + 5)
    seg2_ylist = p(x)
    seg2m, seg2b = list(p)
    #seg2, = plt.plot( x, p(x), label='Segment2', color='g' )
    seg2_line = ( x, p(x))

    #seg1-2 intersetion
    x0 = (seg2b-seg1b)/(seg1m-seg2m)
    y0 = p(x0)
    intersect_1 = (x0,y0)
    #plt.plot(x0,y0, 'ro')
    

    #segment 3
    seg3X, seg3Y = segment_bound(134, 260, 94, 158)
    startline = min(seg3X)
    endline = max(seg3X)
    p_coeff = np.polyfit(seg3X, seg3Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 20, endline + 20)
    seg3m, seg3b = list(p)
    #seg3, = plt.plot( x, p(x), label='Segment3', color='g' )
    seg3_line = ( x, p(x))

    #seg2-3 intersetion
    x0 = (seg3b-seg2b)/(seg2m-seg3m)
    y0 = p(x0)
    intersect_2 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #segment 4
    seg4X =[]
    seg4Y= []
    x,y = segment_bound(295, 322, 286, 325)
    for i in x:
        seg4X.append(i)
    for i in y:
        seg4Y.append(i)
    x1,y1 = segment_bound(322, 360, 213, 271)
    for i in x1:
        seg4X.append(i)
    for i in y1:
        seg4Y.append(i)
    startline = min(seg4X)
    endline = max(seg4X)
    p_coeff = np.polyfit(seg4X, seg4Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 20, endline + 20)
    seg4m, seg4b = list(p)
    #seg4, = plt.plot( x, p(x), label='Segment4', color='g' )
    seg4_line = ( x, p(x))

    #seg1-4 intersetion
    x0 = (seg4b-seg1b)/(seg1m-seg4m)
    y0 = p(x0)
    intersect_5 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #segment 5
    seg5X =[]
    seg5Y= []
    x,y = segment_bound(266, 326, 93, 164)
    for i in x:
        seg5X.append(i)
    for i in y:
        seg5Y.append(i)
    x1,y1 = segment_bound(326, 368, 165, 209)
    for i in x1:
        seg5X.append(i)
    for i in y1:
        seg5Y.append(i)
    startline = min(seg5X)
    endline = max(seg5X)
    p_coeff = np.polyfit(seg5X, seg5Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 20, endline + 20)
    seg5m, seg5b = list(p)
    #seg5, = plt.plot( x, p(x), label='Segment5', color='g' )
    seg5_line = ( x, p(x))

    #seg4-5 intersetion
    x0 = (seg4b-seg5b)/(seg5m-seg4m)
    y0 = p(x0)
    intersect_4 = (x0,y0)
    #plt.plot(x0,y0, 'ro')


    #seg3-5 intersetion
    x0 = (seg3b-seg5b)/(seg5m-seg3m)
    y0 = p(x0)
    intersect_3 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #segment 6
    seg6X, seg6Y = segment_bound(371, 505, 307, 344)
    startline = min(seg6X)
    endline = max(seg6X)
    p_coeff = np.polyfit(seg6X, seg6Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 20, endline + 20)
    seg6m, seg6b = list(p)
    #seg6, = plt.plot( x, p(x), label='Segment6', color='g' )
    seg6_line = ( x, p(x))

    #segment 7
    seg7X, seg7Y = segment_bound(512, 520, 161, 302)
    startline = min(seg7X)
    endline = max(seg7X)
    p_coeff = np.polyfit(seg7X, seg7Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 5, endline + 7)
    seg7m, seg7b = list(p)
    #seg7, = plt.plot( x, p(x), label='Segment7', color='g' )
    seg7_line = ( x, p(x))

    #seg6-7 intersetion
    x0 = (seg6b-seg7b)/(seg7m-seg6m)
    y0 = p(x0)
    intersect_7 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #segment 8
    seg7X, seg7Y = segment_bound(388, 515, 93, 158)
    startline = min(seg7X)
    endline = max(seg7X)
    p_coeff = np.polyfit(seg7X, seg7Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 20, endline + 20)
    seg8m, seg8b = list(p)
    #seg8, = plt.plot( x, p(x), label='Segment8', color='g' )
    seg8_line = ( x, p(x))

    #seg7-8 intersetion
    x0 = (seg7b-seg8b)/(seg8m-seg7m)
    y0 = p(x0)
    intersect_8 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #segment 9
    seg9X =[]
    seg9Y= []
    x,y = segment_bound(327, 384, 94, 160)
    for i in x:
        seg9X.append(i)
    for i in y:
        seg9Y.append(i)
    x1,y1 = segment_bound(285, 325, 165, 207)
    for i in x1:
        seg9X.append(i)
    for i in y1:
        seg9Y.append(i)
    startline = min(seg9X)
    endline = max(seg9X)
    p_coeff = np.polyfit(seg9X, seg9Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 20, endline + 20)
    seg9m, seg9b = list(p)
    #seg9, = plt.plot( x, p(x), label='Segment9', color='g' )
    seg9_line = ( x, p(x))

    #seg8-9 intersetion
    x0 = (seg8b-seg9b)/(seg9m-seg8m)
    y0 = p(x0)
    intersect_9 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #segment 10
    seg10X =[]
    seg10Y= []
    x,y = segment_bound(284, 326, 211, 274)
    for i in x:
        seg10X.append(i)
    for i in y:
        seg10Y.append(i)
    x1,y1 = segment_bound(330, 362, 289, 344)
    for i in x1:
        seg10X.append(i)
    for i in y1:
        seg10Y.append(i)
    startline = min(seg10X)
    endline = max(seg10X)
    p_coeff = np.polyfit(seg10X, seg10Y, 1)
    p = np.poly1d( p_coeff )
    x = np.linspace( startline - 10, endline + 10)
    seg10m, seg10b = list(p)
    #seg10, = plt.plot( x, p(x), label='Segment10', color='g' )
    seg10_line = ( x, p(x))

    #seg9-10 intersetion
    x0 = (seg9b-seg10b)/(seg10m-seg9m)
    y0 = p(x0)
    intersect_10 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #seg6-10 intersetion
    x0 = (seg6b-seg10b)/(seg10m-seg6m)
    y0 = p(x0)
    intersect_6 = (x0,y0)
    #plt.plot(x0,y0, 'ro')

    #plt.pause(0.05)

    lines_list = [seg1_line, seg2_line, seg3_line, seg4_line, seg5_line, seg6_line, seg7_line, seg8_line, seg9_line, seg10_line]
    intersect_list = [intersect_1, intersect_2,intersect_3,intersect_4,intersect_5,intersect_6,intersect_7,intersect_8,intersect_9,intersect_10]

    return lines_list, intersect_list


'''graph the results'''
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=cm.gray)
ax[0].set_title('Input image: MMSE PENTAGONS')
ax[0].set_axis_off()

ax[1].imshow(blank_img)
ax[1].scatter(x_coords,y_coords)
ax[1].set_title('Scatterplot of pixel data')

ax[2].imshow(img)
for i in corners:
    for j in i:
        x = j[0]
        y = j[1]
        ax[2].scatter(x,y,linewidths=5, marker="x", color="r")
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('goodFeaturesToTrack corner algorithm')

##ax[3].imshow(img)
##ax[3].scatter(x,y,linewidths=5, marker="x", color="r")
##ax[3].set_xlim((0, image.shape[1]))
##ax[3].set_ylim((image.shape[0], 0))
##ax[3].set_title('Fixed Corners with Centroid')


ax[6].imshow(img)  
for item in seg_intersect()[0]:  
    ax[6].plot(item[0], item[1] , label='Best Fit Line', color='g')
ax[6].set_xlim((0, image.shape[1]))
ax[6].set_ylim((image.shape[0], 0))
ax[6].set_title('Best Fit line algorithm')

ax[7].imshow(blank_img)
ax[7].scatter(x_coords,y_coords)
for i in seg_intersect()[0]:  
    ax[7].plot(i[0], i[1] , label='Best Fit Line', color='g')
ax[7].set_xlim((0, image.shape[1]))
ax[7].set_ylim((image.shape[0], 0))
ax[7].set_title('Best Fit over Pixel Data')

ax[8].imshow(img)
for i in seg_intersect()[1]:
    ax[8].scatter(i[0],i[1],linewidths=5, marker="x", color="g")
ax[8].set_xlim((0, image.shape[1]))
ax[8].set_ylim((image.shape[0], 0))
ax[8].set_title('Corners with Best Fit')

ax[3].imshow(edges, cmap=cm.gray)
ax[3].set_title('Canny edges [Canny86] algorithm')

ax[4].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[4].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[4].set_xlim((0, pents.shape[1]))
ax[4].set_ylim((pents.shape[0], 0))
ax[4].set_title('Probabilistic Hough Transform')

ax[5].imshow(pents)
for i in lines:
    ax[5].scatter(i[0][0],i[0][1], color="b")
    ax[5].scatter(i[1][0],i[1][1], color="b")
ax[5].set_xlim((0, pents.shape[1]))
ax[5].set_ylim((pents.shape[0], 0))
ax[5].set_title('Corners using Hough')

for a in ax:
    a.set_axis_off()
    a.set_adjustable('box-forced')

fig.canvas.set_window_title('MMSE Pentagon analysis')

plt.tight_layout()
plt.show()

