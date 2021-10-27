import tkinter as tk
from tkinter.constants import ANCHOR, CENTER, NW
from PIL import Image, ImageTk
import cv2
import threading
import os
import numpy as np
from collections import deque
import dlib



class App():
    def __init__(self,window):
        self.panorama_queue=deque()
        self.cnt=0
        self.brightness_temp=100
        self.contrast_temp=0
        threading.Thread.__init__(self)
        self.window=window
        self.window.title("MINI_PROJECT#3")
        self.window.geometry("1200x700+50+50")
        self.window.resizable(False,False)
        
        self.title_label=tk.Label(self.window,text="SNOW_Representation")
        self.title_label.grid(row=0, column=0,columnspan=12)
                
        self.image_canvas=tk.Canvas(self.window,width=432, height=324)
        self.image_canvas.grid(row=1,column=0,rowspan=4)
        self.video_canvas=tk.Canvas(self.window,width=432, height=324)
        self.video_canvas.grid(row=5,column=0,rowspan=4)
                
        self.click_label=tk.Label(self.window,text='take a photo : ')
        self.click_label.grid(row=1,column=1)
        self.click_button=tk.Button(self.window,text="cheese!",command=self.video_to_image)
        self.click_button.grid(row=1,column=2)
        self.click_flag=False
        
        self.blur_label=tk.Label(self.window,text='blur')
        self.blur_label.grid(row=2,column=1)
        self.blur_start_button=tk.Button(self.window,text="blur_start",command=self.blur_bind_activate)
        self.blur_start_button.grid(row=2,column=2)
        self.blur_finish_button=tk.Button(self.window,text="blur_finish",command=self.blur_bind_deactivate)
        self.blur_finish_button.grid(row=2,column=3)
        self.blur_flag=False
        self.blur_sx,self.blur_sy,self.blur_fx,self.blur_fy=None,None,None,None
        
        self.brightness_label=tk.Label(self.window,text='brightness')
        self.brightness_label.grid(row=3,column=1)
        self.brightness10_button=tk.Button(self.window,text='20%',command=lambda : self.brightness_activate(20))
        self.brightness10_button.grid(row=3,column=2)
        self.brightness20_button=tk.Button(self.window,text='40%',command=lambda : self.brightness_activate(40))
        self.brightness20_button.grid(row=3,column=3)
        self.brightness30_button=tk.Button(self.window,text='60%',command=lambda : self.brightness_activate(60))
        self.brightness30_button.grid(row=3,column=4)
        self.brightness40_button=tk.Button(self.window,text='80%',command=lambda : self.brightness_activate(80))
        self.brightness40_button.grid(row=3,column=5)
        self.brightness50_button=tk.Button(self.window,text='100%',command=lambda : self.brightness_activate(100))
        self.brightness50_button.grid(row=3,column=6)
        self.brightness60_button=tk.Button(self.window,text='120%',command=lambda : self.brightness_activate(120))
        self.brightness60_button.grid(row=3,column=7)
        self.brightness70_button=tk.Button(self.window,text='140%',command=lambda : self.brightness_activate(140))
        self.brightness70_button.grid(row=3,column=8)
        self.brightness80_button=tk.Button(self.window,text='160%',command=lambda : self.brightness_activate(160))
        self.brightness80_button.grid(row=3,column=9)
        self.brightness90_button=tk.Button(self.window,text='180%',command=lambda : self.brightness_activate(180))
        self.brightness90_button.grid(row=3,column=10)
        self.brightness_flag=False
        
        self.contrast_label=tk.Label(self.window,text='contrast')
        self.contrast_label.grid(row=4,column=1)
        self.contrast10_button=tk.Button(self.window,text='-80%',command=lambda : self.contrast_activate(-80))
        self.contrast10_button.grid(row=4,column=2)
        self.contrast20_button=tk.Button(self.window,text='-60%',command=lambda : self.contrast_activate(-60))
        self.contrast20_button.grid(row=4,column=3)
        self.contrast30_button=tk.Button(self.window,text='-40%',command=lambda : self.contrast_activate(-40))
        self.contrast30_button.grid(row=4,column=4)
        self.contrast40_button=tk.Button(self.window,text='-20%',command=lambda : self.contrast_activate(-20))
        self.contrast40_button.grid(row=4,column=5)
        self.contrast50_button=tk.Button(self.window,text='0%',command=lambda : self.contrast_activate(0))
        self.contrast50_button.grid(row=4,column=6)
        self.contrast60_button=tk.Button(self.window,text='+20%',command=lambda : self.contrast_activate(20))
        self.contrast60_button.grid(row=4,column=7)
        self.contrast70_button=tk.Button(self.window,text='+40%',command=lambda : self.contrast_activate(40))
        self.contrast70_button.grid(row=4,column=8)
        self.contrast80_button=tk.Button(self.window,text='+60%',command=lambda : self.contrast_activate(60))
        self.contrast80_button.grid(row=4,column=9)
        self.contrast90_button=tk.Button(self.window,text='+80%',command=lambda : self.contrast_activate(80))
        self.contrast90_button.grid(row=4,column=10)
        self.contrast_flag=False
        
        self.liquify_label=tk.Label(self.window,text='liquify')
        self.liquify_label.grid(row=5,column=1)
        self.liquify_start_button=tk.Button(self.window,text="liquify_start",command=self.liquify_bind_activate)
        self.liquify_start_button.grid(row=5,column=2)
        self.liquify_finish_button=tk.Button(self.window,text="liquify_finish",command=self.liquify_bind_deactivate)
        self.liquify_finish_button.grid(row=5,column=3)
        self.liquify_flag=False
        self.liquify_sx,self.liquify_sy,self.liquify_fx,self.liquify_fy=None,None,None,None
        
        self.panorama_label=tk.Label(self.window,text='panorama')
        self.panorama_label.grid(row=6,column=1)
        self.panorama_button=tk.Button(self.window,text='panorama_start',command=self.panorama_activate)
        self.panorama_button.grid(row=6,column=2)
        self.panorama_button=tk.Button(self.window,text='panorama_stop',command=self.panorama_deactivate)
        self.panorama_button.grid(row=6,column=3)
        self.panorama_flag=False
        
        self.face_swap_label=tk.Label(self.window,text='face_swap')
        self.face_swap_label.grid(row=7,column=1)
        self.face_swap_draw_button=tk.Button(self.window,text="draw",command=lambda : self.face_swap_bind_activate(0))
        self.face_swap_draw_button.grid(row=7,column=2)
        self.face_swap_skull_button=tk.Button(self.window,text="skull",command=lambda : self.face_swap_bind_activate(1))
        self.face_swap_skull_button.grid(row=7,column=3)
        self.face_swap_dicafrio_button=tk.Button(self.window,text="dicafrio",command=lambda : self.face_swap_bind_activate(2))
        self.face_swap_dicafrio_button.grid(row=7,column=4)
        self.face_swap_disney_button=tk.Button(self.window,text="disney",command=lambda : self.face_swap_bind_activate(3))
        self.face_swap_disney_button.grid(row=7,column=5)
        self.face_swap_finish_button=tk.Button(self.window,text="face_swap_finish",command=lambda : self.face_swap_bind_deactivate())
        self.face_swap_finish_button.grid(row=7,column=6)
        self.face_swap_flag=False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

        self.cap=cv2.VideoCapture(0)
        self.new_frame=None
        self.delay=15
        self.update()
        self.window.mainloop()
        
    def update(self):
        ret,self.raw=self.cap.read()
        self.raw=cv2.cvtColor(self.raw,cv2.COLOR_BGR2RGB)
        self.frame=cv2.resize(self.raw, dsize=(432,324),interpolation=cv2.INTER_LINEAR)
        
        #face_swap
        if self.face_swap_flag==True:
            try:
                self.frame=self.face_swap(self.character,self.frame)
            except:
                pass
        self.frame=self.contrast(self.frame,0)
        self.photo=ImageTk.PhotoImage(image=Image.fromarray(self.frame))
        self.video_canvas.create_image(0,0,image=self.photo,anchor= NW)
        
        #panorama
        if self.panorama_flag==True:
            self.panorama_queue.append(self.raw)
            cv2.waitKey(5000)
        
 
        
        #capture
        if self.click_flag==True:
            self.origin_frame=self.frame.copy()
            self.fixed_frame=self.frame.copy()
            self.click_flag="captured"
        if self.click_flag=="captured":
            self.fixed_photo=ImageTk.PhotoImage(image=Image.fromarray(self.fixed_frame))
            self.image_canvas.create_image(0,0,image=self.fixed_photo,anchor= NW)   
               
            #blur
            if self.blur_flag==True:
                self.fixed_frame=self.blur(self.fixed_frame)
            #brightness
            if type(self.brightness_flag)==int:                
                self.fixed_frame=self.brightness(self.origin_frame.copy(),self.brightness_flag)
                self.fixed_frame=self.contrast(self.fixed_frame.copy(),self.contrast_temp)
                self.brightness_temp=self.brightness_flag
                self.brightness_flag=True      
            #contrast           
            if type(self.contrast_flag)==int:
                self.fixed_frame=self.brightness(self.origin_frame.copy(),self.brightness_temp)
                self.fixed_frame=self.contrast(self.fixed_frame.copy(),self.contrast_flag)
                self.contrast_temp=self.contrast_flag
                self.contrast_flag=True
            #liquify
            if self.liquify_flag==True:
                    self.fixed_frame=self.liquify(self.fixed_frame)
                    self.liquify_flag=False
        self.window.after(self.delay,self.update)
    def video_to_image(self):
        self.click_flag=True
    def gray(self,img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray
    def edge(self,img):
        edges=cv2.Canny(img,0,255)
        return edges
    def blur(self,img):
        k1=np.array([[3,2,3],
                      [2,1,2],
                      [3,2,3]])*(1/21)
        for i in range(self.blur_sy,self.blur_fy):
            for j in range(self.blur_sx,self.blur_fx):
                img[i,j,0]=sum(sum(img[i-1:i+2,j-1:j+2,0]*(k1))) 
                print(img[i][j][0])               
                img[i,j,1]=sum(sum(img[i-1:i+2,j-1:j+2,1]*(k1)))
                img[i,j,2]=sum(sum(img[i-1:i+2,j-1:j+2,2]*(k1)))                   
        return img  
    def blur_bind_activate(self):
        self.image_canvas.bind("<Button-1>", self.blur_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.blur_mouse_move)
        self.image_canvas.bind("<ButtonRelease-1>", self.blur_mouse_up)
        self.image_canvas.config(cursor="target")
    def blur_bind_deactivate(self):
        self.image_canvas.unbind("<Button-1>")
        self.image_canvas.unbind("<B1-Motion>")
        self.image_canvas.unbind("<ButtonRelease-1>")
        self.image_canvas.config(cursor="")
    def blur_mouse_down(self,evt):
        self.blur_flag=True
        self.blur_sx,self.blur_sy=int(evt.x)-3,int(evt.y)-3
        self.blur_fx,self.blur_fy=int(evt.x)+3,int(evt.y)+3
    def blur_mouse_move(self, evt):
        self.blur_sx,self.blur_sy=int(evt.x)-3,int(evt.y)-3
        self.blur_fx,self.blur_fy=int(evt.x)+3,int(evt.y)+3
    def blur_mouse_up(self,evt):
        self.blur_flag=False
    def brightness_activate(self,percent):
        self.brightness_flag=percent
    def brightness(self,img,percent):
        img=img.astype(np.float32)
        img=img*percent/100
        img=np.clip(img,0,255)
        img=img.astype(np.uint8)
        return img
    def contrast_activate(self,percent):
        self.contrast_flag=percent
    def contrast(self,img,count):
        img=img.astype(np.float32)
        img_b,img_g,img_r=cv2.split(img)
        new_max_b=img_b.max()+count
        new_min_b=img_b.min()-count
        img_b_norm=((img_b-new_min_b)*(new_max_b-new_min_b)/(img_b.max()-img_b.min())+new_min_b)
        img_b_norm=np.clip(img_b_norm,0,255)
        img_b_norm=img_b_norm.astype(np.uint8)

        new_max_g=img_g.max()+count
        new_min_g=img_g.min()-count
        img_g_norm=((img_g-new_min_g)*(new_max_g-new_min_g)/(img_g.max()-img_g.min())+new_min_g)
        img_g_norm=np.clip(img_g_norm,0,255)
        img_g_norm=img_g_norm.astype(np.uint8)

        new_max_r=img_r.max()+count
        new_min_r=img_r.min()-count
        img_r_norm=((img_r-new_min_r)*(new_max_r-new_min_r)/(img_r.max()-img_r.min())+new_min_r)
        img_r_norm=np.clip(img_r_norm,0,255)
        img_r_norm=img_r_norm.astype(np.uint8)  

        img_norm=cv2.merge((img_b_norm,img_g_norm,img_r_norm))
        return img_norm
    def liquify_bind_activate(self):
        self.image_canvas.bind("<Button-1>", self.liquify_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.liquify_mouse_move)
        self.image_canvas.bind("<ButtonRelease-1>", self.liquify_mouse_up)
        self.image_canvas.config(cursor="target")
    def liquify_bind_deactivate(self):
        self.image_canvas.bind("<Button-1>")
        self.image_canvas.bind("<B1-Motion>")
        self.image_canvas.bind("<ButtonRelease-1>")
        self.image_canvas.config(cursor="")
    def liquify_mouse_down(self,evt):  
        self.liquify_sx,self.liquify_sy=int(evt.x),int(evt.y)
    def liquify_mouse_move(self,evt):
        self.liquify_fx,self.liquify_fy=int(evt.x),int(evt.y)
    def liquify_mouse_up(self,evt):
        self.liquify_flag=True
    def liquify(self,img,half=30,mul=2) :
        # 대상 영역 좌표와 크기 설정
        img=cv2.resize(img,dsize=(1296,972),interpolation=cv2.INTER_LINEAR)
        tan_x=self.liquify_fx-self.liquify_sx
        tan_y=self.liquify_fy-self.liquify_sy
        self.liquify_sx=self.liquify_sx*3
        self.liquify_sy=self.liquify_sy*3    
        self.liquify_fx=self.liquify_fx*3
        self.liquify_fy=self.liquify_fy*3
        if(tan_x==0):
            img=cv2.resize(img,dsize=(432,324),interpolation=cv2.INTER_LINEAR)
            return img
        tan=abs(int(mul*tan_y/tan_x))
        if tan_x<0:
            self.liquify_fx=self.liquify_sx-mul
        else:
            self.liquify_fx=self.liquify_sx+mul
        if tan_y<0:
            self.liquify_fy=self.liquify_sy-tan
        else:
            self.liquify_fy=self.liquify_sy+tan


        x, y, w, h = self.liquify_sx-half, self.liquify_sy-half, half*2, half*2
        # 관심 영역 설정
        roi = img[y:y+h, x:x+w].copy()
        out = roi.copy()

        # 관심영역 기준으로 좌표 재 설정
        offset_cx1,offset_cy1 = self.liquify_sx-x, self.liquify_sy-y
        offset_cx2,offset_cy2 = self.liquify_fx-x, self.liquify_fy-y

        # 변환 이전 4개의 삼각형 좌표
        tri1 = [[ (0,0), (w, 0), (offset_cx1, offset_cy1)], # 상,top
                [ [0,0], [0, h], [offset_cx1, offset_cy1]], # 좌,left
                [ [w, 0], [offset_cx1, offset_cy1], [w, h]], # 우, right
                [ [0, h], [offset_cx1, offset_cy1], [w, h]]] # 하, bottom

        # 변환 이후 4개의 삼각형 좌표
        tri2 = [[ [0,0], [w,0], [offset_cx2, offset_cy2]], # 상, top
                [ [0,0], [0, h], [offset_cx2, offset_cy2]], # 좌, left
                [ [w,0], [offset_cx2, offset_cy2], [w, h]], # 우, right
                [ [0,h], [offset_cx2, offset_cy2], [w, h]]] # 하, bottom


        for i in range(4):
            # 각각의 삼각형 좌표에 대해 어핀 변환 적용
            matrix = cv2.getAffineTransform( np.float32(tri1[i]), \
                                             np.float32(tri2[i]))
            warped = cv2.warpAffine( roi.copy(), matrix, (w, h), \
                None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            # 삼각형 모양의 마스크 생성
            mask = np.zeros((h, w), dtype = np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255,255,255))

            # 마스킹 후 합성
            warped = cv2.bitwise_and(warped, warped, mask=mask)
            out = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(mask))
            out = out + warped

        # 관심 영역을 원본 영상에 합성
        img[y:y+h, x:x+w] = out
        img=cv2.resize(img,dsize=(432,324),interpolation=cv2.INTER_LINEAR)
        return img 
    def remap_barrel(self,img):
        k1,k2,k3=0.5,0.2,0.0
        rows,cols=img.shape[:2]
        mapy,mapx=np.indices((rows,cols),dtype=np.float32)
        mapx=2*mapx/(cols-1)-1
        mapy=2*mapy/(rows-1)-1
        r,theta=cv2.cartToPolar(mapx,mapy)
        
        ru=r*(1+k1*(r**2)+k2*(r**4)+k3*(r**6))
        mapx,mapy=cv2.polarToCart(ru,theta)
        mapx=((mapx+1)*cols-1)/2
        mapy=((mapy+1)*rows-1)/2
        distored=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        return distored
    def panorama_activate(self):
        self.panorama_flag=True           
    def panorama_deactivate(self):
        self.panorama_flag=False
        self.panorama_window=tk.Toplevel()
        title_label=tk.Label(self.panorama_window,text="panorama_window")
        title_label.grid(row=0,column=0)
        self.panorama_canvas=tk.Canvas(self.panorama_window,width=1000, height=400)
        self.panorama_canvas.grid(row=1,column=0)
        panorama_save_label=tk.Button(self.panorama_window,text="save",command=self.save_panorama)
        panorama_save_label.grid(row=2,column=0)
        print(len(self.panorama_queue))
        now=self.panorama_queue.popleft()
        cv2.imshow('N',now)
        while(len(self.panorama_queue)!=0):
            new=self.panorama_queue.popleft()
            now=self.panorama(now,new)  
        cv2.imshow('P',now)
        panorama_img=ImageTk.PhotoImage(image=Image.fromarray(now))
        self.panorama_canvas.create_image(0,0,image=panorama_img,anchor= NW)
    def save_panorama(self):
        pass       
    def panorama(self,imgL,imgR):
        hl,wl=imgL.shape[:2]
        hr,wr=imgR.shape[:2]
        grayL=cv2.cvtColor(imgL,cv2.COLOR_RGB2GRAY)
        grayR=cv2.cvtColor(imgR,cv2.COLOR_RGB2GRAY)
        
        descriptor=cv2.xfeatures2d.SIFT_create()
        (kpsL,featuresL)=descriptor.detectAndCompute(imgL,None)
        (kpsR,featuresR)=descriptor.detectAndCompute(imgR,None)
        matcher=cv2.DescriptorMatcher_create("BruteForce")
        matches=matcher.knnMatch(featuresR,featuresL,2)
        good_matches=[]
        for m in matches:
            if len(m)==2 and m[0].distance<m[1].distance*0.75:
                good_matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(good_matches)>4:
            ptsL=np.float32([kpsL[i].pt for (i,_) in good_matches])
            ptsR=np.float32([kpsR[i].pt for (_,i) in good_matches])
            mtrx,status=cv2.findHomography(ptsR,ptsL,cv2.RANSAC,4.0)
            panorama=cv2.warpPerspective(imgR,mtrx,(wr+wl,hr))
            panorama[0:hl,0:wl]=imgL
        else:
            panorama=imgL
        return panorama
    def face_swap_bind_activate(self,choose):
        if choose==0:
            self.character=cv2.imread("./data/draw.jpg")
        elif choose==1:
            self.character=cv2.imread("./data/skull.jpg")
        elif choose==2:
            self.character=cv2.imread("./data/dicafrio.jpg")
        elif choose==3:
            self.character=cv2.imread("./data/disney.jpg")
        self.face_swap_flag=True      
    def face_swap_bind_deactivate(self):
        self.face_swap_flag=False   
    def getPoints(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray)
        points = []
        for rect in rects:
            shape = self.predictor(gray, rect)
            for i in range(68):
                part = shape.part(i)
                points.append((part.x, part.y))
        return points    
    def getTriangles(self,img, points):
        w,h = img.shape[:2]
        subdiv = cv2.Subdiv2D((0,0,w,h));
        subdiv.insert(points) 
        triangleList = subdiv.getTriangleList();
        triangles = []
        for t in triangleList:        
            pt = t.reshape(-1,2)
            if not (pt < 0).sum() and not (pt[:, 0] > w).sum() \
                                  and not (pt[:, 1] > h).sum(): 
                indice = []
                for i in range(0, 3):
                    for j in range(0, len(points)):                    
                        if(abs(pt[i][0] - points[j][0]) < 1.0 \
                            and abs(pt[i][1] - points[j][1]) < 1.0):
                            indice.append(j)    
                if len(indice) == 3:                                                
                    triangles.append(indice)
        return triangles
    def warpTriangle(self,img1, img2, pts1, pts2):
        x1,y1,w1,h1 = cv2.boundingRect(np.float32([pts1]))
        x2,y2,w2,h2 = cv2.boundingRect(np.float32([pts2]))

        roi1 = img1[y1:y1+h1, x1:x1+w1]
        roi2 = img2[y2:y2+h2, x2:x2+w2]

        offset1 = np.zeros((3,2), dtype=np.float32)
        offset2 = np.zeros((3,2), dtype=np.float32)
        for i in range(3):
            offset1[i][0], offset1[i][1] = pts1[i][0]-x1, pts1[i][1]-y1
            offset2[i][0], offset2[i][1] = pts2[i][0]-x2, pts2[i][1]-y2

        mtrx = cv2.getAffineTransform(offset1, offset2)
        warped = cv2.warpAffine( roi1, mtrx, (w2, h2), None, \
                            cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101 )

        mask = np.zeros((h2, w2), dtype = np.uint8)
        cv2.fillConvexPoly(mask, np.int32(offset2), (255))

        warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
        roi2_masked = cv2.bitwise_and(roi2, roi2, mask=cv2.bitwise_not(mask))
        roi2_masked = roi2_masked + warped_masked
        img2[y2:y2+h2, x2:x2+w2] = roi2_masked
    def face_swap(self,img1,img2):
        img_draw = img2.copy()
        # 각 이미지에서 얼굴 랜드마크 좌표 구하기--- ⑥ 
        points1 = self.getPoints(img1)
        points2 = self.getPoints(img2)
        # 랜드마크 좌표로 볼록 선체 구하기 --- ⑦
        hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
        hull1 = [points1[int(idx)] for idx in hullIndex]
        hull2 = [points2[int(idx)] for idx in hullIndex]
        # 볼록 선체 안 들로네 삼각형 좌표 구하기 ---⑧ 
        triangles = self.getTriangles(img2, hull2)

        # 각 삼각형 좌표로 삼각형 어핀 변환 ---⑨    
        for i in range(0, len(triangles)):
            t1 = [hull1[triangles[i][j]] for j in range(3)]
            t2 = [hull2[triangles[i][j]] for j in range(3)]
            self.warpTriangle(img1, img_draw, t1, t2)
        # 볼록선체를 마스크로 써서 얼굴 합성 ---⑩
        mask = np.zeros(img2.shape, dtype = img2.dtype)  
        cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))
        r = cv2.boundingRect(np.float32([hull2]))    
        center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
        output = cv2.seamlessClone(np.uint8(img_draw), img2, mask, center, \
                                    cv2.NORMAL_CLONE)
        return output

App(tk.Tk())
