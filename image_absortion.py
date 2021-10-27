import cv2
import numpy as np
a=cv2.imread("./data/face_brightness-40.png")
b=cv2.imread("./data/face_brightness+40.png")
c=cv2.imread("./data/face_contrast-40.png")
d=cv2.imread("./data/face_contrast+40.png")

ah,aw,_=a.shape
bh,bw,_=b.shape
ch,cw,_=c.shape
dh,dw,_=d.shape

n1=np.zeros((ah,aw+bw,3),np.uint8)
n2=np.zeros((ch,cw+dw,3),np.uint8)

for i in range(ah):
    for j in range(aw):
        for k in range(3):
            n1[i][j][k]=a[i][j][k]
for i in range(bh):
    for j in range(bw):
        for k in range(3):
            try:
                n1[i][j+aw][k]=b[i][j][k]
            except:        
                pass
for i in range(ch):
    for j in range(cw):
        for k in range(3):
            n2[i][j][k]=c[i][j][k]
for i in range(dh):
    for j in range(dw):
        for k in range(3):
            try:
                n2[i][j+cw][k]=d[i][j][k]
            except:
                pass
            
cv2.imwrite("./data/brightness.png",n1)
cv2.imwrite("./data/contrast.png",n2)