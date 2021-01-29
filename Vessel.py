import cv2
import numpy as np
from matplotlib import pyplot as plt


def vessel_seg(img):
    
    image=cv2.imread(img)
    b,g,r = cv2.split(image)   #split the image into individual channels (blue,green or red)
    cv2.imwrite("static/"+img[:-4]+"_green_channel.jpg",g)
    cv2.imwrite("static/"+img[:-4]+"_red_channel.jpg",r)
    cv2.imwrite("static/"+img[:-4]+"_blue_channel.jpg",b)
    # cv2.imwrite("static/blue_channel.jpg",b)
    # cv2.imwrite("static/green_channel.jpg",g)
    # cv2.imwrite("static/red_channel.jpg",r)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #cv2.createCLAHE is used as adaptive histogram equalization
    mg_clahe = clahe.apply(g)


    
    r1 = cv2.morphologyEx(mg_clahe, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv2.subtract(R3,mg_clahe)
    f5 = clahe.apply(f4)


    # cv2.imwrite("hbhj.jpg",f5)
    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)   #remove low intensity objects by clipping anything with less than 15 value brightness
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        if cv2.contourArea(cnt) <= 175:
            cv2.drawContours(mask, [cnt], -1, 0, -1)   # contains all contours having Area of less than 175 pixels


    im = cv2.bitwise_and(f5, f5, mask=mask)    #Using bitwise_and operator all small contours are removed
    
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	


    image_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    # cv2.imwrite("hbj.jpg",image_eroded)
    xcontours, xhierarchy = cv2.findContours(image_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #Removing small unwanted contour which are not part of blood vessels
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(image_eroded,image_eroded,mask=xmask)
    final = cv2.bitwise_not(finimage)

    cv2.imwrite("static/"+img[:-4]+"_blood_vessel.jpg",final)
    # cv2.imwrite("static/"+img[:-4]+"_blood_vessel.jpg",final)

# fig = plt.figure(figsize=(20, 20))
# fig.add_subplot(1, 3, 1)
# plt.imshow(np.asarray(image))
# plt.title("Original Image")
# plt.axis("off")
# plt.subplot(1, 3, 2)
# plt.imshow(np.asarray(final))
# plt.title("Ground Truth")
# plt.axis("off")
# plt.axis("off")
