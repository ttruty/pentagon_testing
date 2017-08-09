# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2, os, time, math, itertools, random
import numpy as np
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

path = r'E:\pent_python\training_mmse_pentagons'
model_path =  os.path.join(path, "models", "svm.model")


class corrections():
    '''
    holder for the x,y hieght width corection from detections
    '''
    x = 0
    y = 0
    w = 0
    h = 0

class target_focus():
    '''
    holder to the target sweep x,y,w,h method of detection
    '''
    x = []
    y = []
    w = []
    h = []
    
class detectionOutput():
    '''
    output to load into image or xls
    '''
    orgShape = (0,0)
    harris_corners = []
    harris_lines = []
    shi_corners = []
    shi_lines = []

def line_points(start, end):
        "Bresenham's line algorithm"
        x0, y0 = start
        x1, y1 = end
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append((x, y))
        return points_in_line

def cornerMeths(im, detections, corners, quality, distance, line_threshold):
    '''
    Methods for the corner detections including the corrections for the scaling factors
    '''
    x_min = []  # Make container box with all features
    y_min = []
    x_max = []
    y_max = []
    for (x, y, _, w, h) in detections:
        x_min.append(x)
        y_min.append(y)
        x_max.append(x+w)
        y_max.append(y+h)    
    x = min(x_min)
    y = min(y_min)
    w = max(x_max)
    h = max(y_max)
    corrections.x = x
    corrections.y = y
    corrections.w = w
    corrections.h = h

    imCrop = im[y:h, x:w]
    #cv2.imshow("cropped", imCrop)
    imageShape = im.shape
    detectionOutput.orgShape = imageShape
    cropShape = imCrop.shape
    #print("original image size:", im.shape)
    #print("Crop Shape: ", imCrop.shape)

    color_im = cv2.cvtColor(imCrop,cv2.COLOR_GRAY2RGB)


    shi_corners = shiCorners(color_im, corners, quality, distance)
    shi_lines = connectLines(im, shi_corners, line_threshold)

    return shi_corners

    #correctLine(im, shi_lines[0]
def detection(img):
    '''
    The detections methond using the HOG algorithm
    '''
    
    clf = joblib.load(model_path) #prediciton method using SVM
    kernel = np.ones((5,5),np.uint8)
    #erosion = cv2.erode(im_reshape,kernel,iterations = 1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    im_reshape = cv2.resize(opening, (128,128))
    
    fd, _ = hog(im_reshape,9, (8,8), (3,3), visualise = True, transform_sqrt=True)           
    pred = clf.predict(fd)
    #print(pred)
    #print(clf.decision_function(fd))
    return clf.decision_function(fd)


def find_contours(path, corners, quality, distance, detection_threshold):
    '''
    cv2 contour finding to get the shapes on page according to the black pixels
    '''
    found_dets = []
    
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    clone = img.copy()

    group_x = []
    group_y = []
    
    blur = cv2.GaussianBlur(im,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##    cv2.imshow("thresh", th3)
##    cv2.waitKey()
    
    edges = cv2.Canny(img,0,255,apertureSize = 3) # Canny image 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) # create the kernle
    #kernel = np.ones((5,5),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=5)
    #erosion = cv2.erode(im_reshape,kernel,iterations = 1)
    #opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
##    cv2.imshow('Edges', edges)
    im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # find each contour
    cv2.drawContours(clone, contours, -1, (0,255,0), 3) #draw all contours
    #cv2.imshow("ALL CONT", clone)
    found_cnts = []
    rectangles = []
    cv2.drawContours(edges, contours, -1, (0,255,255), 3)
    for cnt in contours:
        # get the perimeter circularity of the contours
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull,True)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        cv2.drawContours(img, cnt, -1, (0,255,255), 3)
        if perimeter != 0:
                x,y,w,h = cv2.boundingRect(cnt)
                if (h*w) > 400: # minimum area of the contour bounding box                   
                    feature = im[y:y + h, x:x + w]
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    cd = detection(feature)
                    if (cd > detection_threshold):
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                        found_dets.append((x, y, cd, w, h))
    return img, found_dets

def connectLines(img,corners, line_threshold):
    '''
    connect the found corners in image with lines
    check if the lines are over black pixels
    '''
## img = imCrop, corners = c_corners
    color_im = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    edges = cv2.Canny(img,0,255,apertureSize = 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) 
    edges = cv2.dilate(edges, kernel, iterations=5)
##    cv2.imshow('dilated edges',edges)
    org_im = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    color_im = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    copy = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    edges=edges.astype(np.uint8)
                 
    line_count = 0
    found_lines = []
    bad_lines = []
    lines = itertools.combinations(corners,2) # create all possible lines
    line_img = np.ones_like(color_im)*255 # white image to draw line markings on
    for line in lines: # loop through each line
        bin_line = np.zeros_like(edges) # create a matrix to draw the line in
        start, end = line # grab endpoints
        points = line_points(start, end)
        cv2.line(bin_line, tuple(start), tuple(end), color=255, thickness=1) # draw line
        conj = (edges/255 + bin_line/255) # create agreement image
        n_agree = np.sum(conj==2)
        n_wrong = np.sum(conj==1)
        
        if n_agree/(len(points)) > line_threshold: # high agreements vs disagreements
            #cv2.line(org_im, tuple(start), tuple(end), color=[0,200,0], thickness=2)
            line = [start, end]
            found_lines.append(line)
            line_count += 1
        if n_agree/(len(points)) < .95 and n_agree/(len(points)) > .85: # high agreements vs disagreements
            #cv2.line(org_im, tuple(start), tuple(end), color=[155,0,0], thickness=2)
            line = [start, end]
            bad_lines.append(line)
    #print('number of found lines', len(found_lines))
    return found_lines, bad_lines, line_count


def correctLine(img, lines):
    bin_line = np.zeros_like(img) # create a matrix to draw the line in
    for i in lines:
        startX,startY = i[0]
        stopX, stopY = i[1]
        fix_startX =  startX + corrections.x
        fix_startY = startY + corrections.y
        fix_stopX = stopX + corrections.x
        fix_stopY = stopY + corrections.y
        start = (fix_startX, fix_startY)
        end = (fix_stopX, fix_stopY)
        cv2.line(bin_line, start, end, color=255, thickness=5)
    #cv2.imshow('bin line', bin_line)

def shiCorners(img, corners, quality, distance):
    '''
    Corners with Shi Tomais corner detection
    '''
    fixed_corners = []
    bin_inv = cv2.bitwise_not(img) # flip image colors
    bin_inv = cv2.cvtColor(bin_inv, cv2.COLOR_BGR2GRAY) # make one channel
    goodcorners = cv2.goodFeaturesToTrack(bin_inv,corners,quality,distance)
    goodcorners = np.int0(goodcorners)

    for i in list(goodcorners):
        X = i[0][0]+ corrections.x
        Y = i[0][1]+ corrections.y
        fixed_corners.append([X, Y])       
    return fixed_corners

def visualizeImg(img, corners, lines, method):
    '''
    This is the code to show images
    '''
    cv2.imshow('Original Image', img)
    # Correct corners for org image
    org_corners = []
    img1 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for i in list(corners):
        cropX,cropY = i
        rndX = cropX #+ corrections.x
        rndY = cropY #+ corrections.y
        org_corners.append([rndX, rndY])
    
    for i in org_corners:
        cv2.circle(img1,(i[0],i[1]),5,255,-1)
    cv2.imshow(method + ' Corners on original', img1)
    
    # Correct lines for harrsi corners on org image
    org_lines = []
    img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for i in lines:
        startX,startY = i[0]
        stopX, stopY = i[1]
        fix_startX =  startX #+ corrections.x
        fix_startY = startY #+ corrections.y
        fix_stopX = stopX #+ corrections.x
        fix_stopY = stopY #+ corrections.y
        start = (fix_startX, fix_startY)
        end = (fix_stopX, fix_stopY)
        org_lines.append([start,end])
        cv2.line(img2, start, end, color=[0,200,0], thickness=2)
    cv2.imshow(method + 'Lines on original', img2)
