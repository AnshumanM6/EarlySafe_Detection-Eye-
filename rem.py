import cv2
import os


def ex_clean(image):
    i1=cv2.imread('static/'+image[:-4]+'_od.jpg',0)
    i2=cv2.imread('static/'+image[:-4]+'_ex.jpg',0)
    for i in range(400):
        for j in range(400):
            if i1[i][j]>0:
                i2[i][j]=0
    cv2.imwrite('static/'+image[:-4]+'_ex.jpg',i2)


def hem_clean(image):
    i1=cv2.imread(image,0)
    i1=cv2.resize(i1,(400,400))
    i2=cv2.imread('static/'+image[:-4]+'_hem.jpg',0)
    for i in range(400):
        for j in range(400):
            if i1[i][j]<1:
                i2[i][j]=0
    cv2.imwrite('static/'+image[:-4]+'_hem.jpg',i2)
