import cv2
import numpy as np 
from matplotlib import pyplot as plt 
# LOading the image
img = cv2.imread('road1.jpg')
# Converting to RBG format to display in pyplot
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Finding height and width
height = img.shape[0]
width = img.shape[1]
# defining variables for region of interest
roi_vertices = [
    (0,height),
    (width/2.5,height/2),
    (width,height-100)
]
# Creating a mask
def roi_mask(img,vertices):
    mask = np.zeros_like(img)
  #  channel_count = img.shape[2]
  #  match_mask_color = (255,)* channel_count #this is for matching the mask color
    match_mask_color = 255 #this is done becaz we are passing gray scale image which has only one color so that is why we are not  detecting the channel count also
    cv2.fillPoly(mask,vertices,match_mask_color) # this is for filling the polygon the colours in the image and storing it in mask
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

# Converting to gray scale
gray_crop = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# Canny edge detection
edge = cv2.Canny(gray_crop,100,200,apertureSize=3)

# Here we would crop the image to dimension which is only necesary for us
cropped_image = roi_mask(edge,np.array([roi_vertices],np.int32))

# Using this cropped image we would execute the lane detection algorithm using prob_hough
# adding the first two steps before region of interest mask to avoid detecting mask edges( it is from line 27 to 30)
#Finding lines using Prob_HOugh
lines = cv2.HoughLinesP(cropped_image,rho=6,theta=np.pi/60,threshold=160,lines=np.array([]),minLineLength=40,maxLineGap=25)

# Iterating through the lines
def draw_lines(img1,lines1):
    img1 = np.copy(img1)
    blank_img = np.zeros((img1.shape[0],img1.shape[1],3),dtype=np.uint8)
    for line in lines1:
        #Finding the coordinates to plot the line
        for x1,y1,x2,y2 in line:
            #Drawing the line
            cv2.line(blank_img,(x1,y1),(x2,y2),(0,255,0),5)
    img1 = cv2.addWeighted(img1,0.8,blank_img,1,0.0)
    return img1

img_with_lines = draw_lines(img,lines)

plt.imshow(img_with_lines)
plt.show()

cv2.destroyAllWindows()
