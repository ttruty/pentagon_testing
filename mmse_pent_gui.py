# import the necessary packages
from tkinter import * 
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import cv2, os, datetime
import numpy as np
import corner_dets_methods


corner_num = 20
quality = 0.1
distance = 15
detection_threshold = 0.20
line_threshold= .95

def apply_cmd():
    global image
    global corner_num
    corner_num = int(corner_count.get())

    global quality
    quality = quality_count.get()
    
    global distance
    distance = int(distance_count.get())

    global detection_threshold
    detection_threshold = det_var.get()

    global line_threshold
    line_threshold = line_var.get()
    
    detections, found_dets = detection_funct()
    load_image(image,detections, found_dets)

def save_cmd():
    with open( os.path.basename(path)+".txt", "w") as text_file:
        text_file.write("MMSE Pentagon GUI Version=1\n")
        text_file.write("Process date = {0}\n".format(datetime.datetime.now()))
        text_file.write("corner count =  {0}\n".format(corner_total))
        text_file.write("line count =  {0}\n".format(line_total))
        text_file.write("min corner count =  {0}\n".format(corner_num))
        text_file.write("min quality =  {0}\n".format(quality))
        text_file.write("distance =  {0}\n".format(distance))
        text_file.write("detection_threshold =  {0}\n".format(detection_threshold))
        text_file.write("line_threshold =  {0}\n".format(line_threshold))
        text_file.write("corner list =  {0}\n".format(shi_c))
        text_file.write("lines list =  {0}\n".format(shi_lines))
        text_file.write("multipent =  {0}\n".format(multipent.get()))
        text_file.write("flag =  {0}\n".format(flag_var.get()))
        
            
    
def apply_corners(image, corner_list):
    for i in corner_list:
            cv2.circle(image,(i[0],i[1]),5,255,-1)

def apply_lines(image, lines):
    for i in lines:
        startX,startY = i[0]
        stopX, stopY = i[1]
        fix_startX =  startX #+ corrections.x
        fix_startY = startY #+ corrections.y
        fix_stopX = stopX #+ corrections.x
        fix_stopY = stopY #+ corrections.y
        start = (fix_startX, fix_startY)
        end = (fix_stopX, fix_stopY)
        cv2.line(image, start, end, color=[0,200,0], thickness=2)

def detection_funct():
    global path
    detections, found_dets = corner_dets_methods.find_contours(path, corner_num, quality ,distance, detection_threshold)
    return detections, found_dets   

def select_image():
    # grab a reference to the image panels
    # open a file chooser dialog and allow the user to select an input
    # image
    
    global image, path, file_name
    path = tkinter.filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        file_name = os.path.basename(path)
        file_label = Label(root, text= file_name)
        file_label.grid(row=0, column=0, columnspan=2)
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        clone = image.copy()
        line_img = image.copy()
        detections, found_dets = detection_funct()
        load_image(image,detections, found_dets)

def corner_details(image, corner_image, line_image, detections):
    global shi_c, shi_lines
    shi_c = corner_dets_methods.cornerMeths(image, [detections], corner_num, quality, distance, line_threshold)
    shi_lines, _, line_count = corner_dets_methods.connectLines(image, shi_c, line_threshold)                      
    apply_corners(corner_image,shi_c)
    apply_lines(line_image, shi_lines)
    corner_count = len(shi_c)
    return corner_count, line_count
    
def load_image(image, detections, found_dets):
    global panelA, panelB, panelC, panelD, multipent, corner_total, line_total
    
    clone = image.copy()
    line_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    line_img = np.zeros_like(clone)
    if found_dets != []:
        if len(found_dets) > 1 and multipent.get() == 0:
            found_pent = max(found_dets, key=lambda x: x[3]*x[4])
            corner_total, line_total = corner_details(gray, clone, line_img,found_pent)            
        elif len(found_dets) > 1 and multipent.get() == 1:
            corner_total = 0
            line_total = 0
            for i in found_dets:
                corner_count, line_count = corner_details(gray, clone, line_img, i)
                corner_total += corner_count
                line_total += line_count              
        else:
            found_pent = found_dets[0]
            corner_total, line_total = corner_details(gray, clone, line_img,found_pent) 

    corner_count_label = Label(root, text='Corner Count: ' + str(corner_total))
    corner_count_label.grid(row=7, column=0)

    line_count_label = Label(root, text='Line Count: ' + str(line_total))
    line_count_label.grid(row=8, column=0)
    
    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
    line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
    dets = cv2.cvtColor(detections, cv2.COLOR_BGR2RGB)

    # convert the images to PIL format...
    image = Image.fromarray(image)
    image = image.resize((250, 400), Image.ANTIALIAS)
    clone = Image.fromarray(clone)
    clone = clone.resize((250, 400), Image.ANTIALIAS)
    dets = Image.fromarray(dets)
    dets = dets.resize((250, 400), Image.ANTIALIAS)
    line_img = Image.fromarray(line_img)
    line_img = line_img.resize((250, 400), Image.ANTIALIAS)
    
    # ...and then to ImageTk format
    image = ImageTk.PhotoImage(image)
    clone = ImageTk.PhotoImage(clone)
    dets = ImageTk.PhotoImage(dets)
    line_img = ImageTk.PhotoImage(line_img)

    # if the panels are None, initialize them
    if panelA is None or panelB is None:
        # the first panel will store our original image
        panelA = Label(image=image)
        panelA.image = image
        panelA.grid(row=1, column=3, rowspan = 20, padx=10)

        # while the second panel will store the detection map
        panelB = Label(image=dets)
        panelB.image = dets
        panelB.grid(row=1, column=4, rowspan = 20, padx=10)

        panelC = Label(image=clone)
        panelC.image = clone
        panelC.grid(row=1, column=5, rowspan = 20, padx=10)

        panelD = Label(image=line_img)
        panelD.image = line_img
        panelD.grid(row=1, column=6, rowspan = 20, padx=10)

    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image)
        panelB.configure(image=dets)
        panelC.configure(image=clone)
        panelD.configure(image=line_img)
        
        panelA.image = image
        panelB.image = dets
        panelC.image = clone
        panelD.image = line_img
        

# initialize the window toolkit along with the image panels
root = Tk()
root.wm_title("MMSE Pentagon")
#root.attributes('-fullscreen', True)

panelA = None
panelB = None
panelC = None
panelD = None
 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI

btn = Button(root, text="Select an image", command=select_image)
btn.grid(row=21, column=0, columnspan=2, sticky=W+S,)

image_label = Label(root, text="Orginal Image")
image_label.grid(row=0, column=3)

det_label = Label(root, text="Detection Image")
det_label.grid(row=0, column=4)

corner_label = Label(root, text="Corner Image")
corner_label.grid(row=0, column=5)

corner_label = Label(root, text="Line Image")
corner_label.grid(row=0, column=6)

c_label = Label(root, text="Corner count")
c_label.grid(row=1, column=0)
corner_count = DoubleVar()
c_scale = Scale(orient='horizontal', from_=0, to=99, variable = corner_count )
c_scale.grid(row=1, column=1 , sticky=N+W)
c_scale.set(corner_num)

q_label = Label(root, text="Min Quality")
q_label.grid(row=2, column=0)
quality_count = DoubleVar()
q_scale = Scale(orient='horizontal', from_=0.01, to=1.00, resolution=0.01, variable = quality_count)
q_scale.grid(row=2, column=1, sticky=N+W)
q_scale.set(quality)

q_label = Label(root, text="Min distance")
q_label.grid(row=3, column=0)
distance_count = DoubleVar()
d_scale = Scale(orient='horizontal', from_=1, to=50, variable = distance_count)
d_scale.grid(row=3, column=1, sticky=N+W)
d_scale.set(distance)

detection_label = Label(root, text="Detection Threshold")
detection_label.grid(row=4, column=0)
det_var = DoubleVar()
det_scale = Scale(orient='horizontal', from_=-3.000, to=3.000, resolution=0.001, variable = det_var)
det_scale.grid(row=4, column=1, sticky=N+W)
det_scale.set(detection_threshold)

line_label = Label(root, text="Line Threshold")
line_label.grid(row=5, column=0)
line_var = DoubleVar()
line_scale = Scale(orient='horizontal', from_=0.01, to=1.00, resolution=0.01, variable = line_var)
line_scale.grid(row=5, column=1, sticky=N+W)
line_scale.set(line_threshold)

export_btn = Button(root, text="Save", command=save_cmd)
export_btn.grid(row=10, column=0)

flag_var = IntVar()
flag_check = Checkbutton(root, text="Flag Page", variable=flag_var)
flag_check.grid(row=11, column=0, sticky="w")

apply_btn = Button(root, text="Apply", command=apply_cmd)
apply_btn.grid(row=6, column=0, columnspan=2)

multipent = IntVar()
multi_check = Checkbutton(root, text="Multiple pentagons", variable=multipent)
multi_check.grid(row=13, column=0)

# kick off the GUI
root.mainloop()
