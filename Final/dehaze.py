# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument('--type', required=True)
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output')
args = parser.parse_args()

CLIP = lambda x: np.uint8(max(0, min(x, 255)))
AtmosphericLight_Y = 0
AtmosphericLight = np.zeros(3)

class Dehazing:
    def __init__(self, img_input):
        self.img_input = img_input
        self.imgY = cv2.cvtColor(img_input, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        self.AtmosphericLight = AtmosphericLight
        self.AtmosphericLight_Y = AtmosphericLight_Y
        self.width = self.img_input.shape[1]
        self.height = self.img_input.shape[0]
        self.pfTransmission = np.zeros(img_input.shape[:2])
    
    def AirLightEstimation(self, origin, height, width):
        UpperLeft  = self.img_input[origin[0]:origin[0]+int(round(height/2)), origin[1]:origin[1]+int(round(width/2))]
        UpperRight = self.img_input[origin[0]:origin[0]+int(round(height/2)), origin[1]+int(round(width/2)):origin[1]+width]
        LowerLeft  = self.img_input[origin[0]+int(round(height/2)):origin[0]+height, origin[1]:origin[1]+int(round(width/2))]
        LowerRight = self.img_input[origin[0]+int(round(height/2)):origin[0]+height, origin[1]+int(round(width/2)):origin[1]+width]
        
        if height*width > 200:
            maxVal = 0
            idx = -1
            for i, blk in enumerate([UpperLeft, UpperRight, LowerLeft, LowerRight]):
                D = np.mean(blk) - np.std(blk)
                if D > maxVal:
                    maxVal = D
                    idx = i 
            self.AirLightEstimation(( origin[0]+int(idx/2)*int(round(height/2)),
                                      origin[1]+int(idx%2)*int(round(width/2))),
                                      int(round(height/2)), int(round(width/2)))
        else:
            global AtmosphericLight, AtmosphericLight_Y
            minDist = 1e10
            for i in range(height):
                for j in range(width):
                    Dist = np.linalg.norm(self.img_input[origin[0]+i,origin[1]+j,:] - np.array([255,255,255]))
                    if Dist < minDist:
                        minDist = Dist
                        self.AtmosphericLight = self.img_input[origin[0]+i, origin[1]+j,:]
                        ## RGB -> Y
                        self.AtmosphericLight_Y = int((self.AtmosphericLight[2]*0.299 + self.AtmosphericLight[1]*0.587 + self.AtmosphericLight[0]*0.114))
                        AtmosphericLight = self.AtmosphericLight
                        AtmosphericLight_Y = self.AtmosphericLight_Y

            ## renew airlight when abrupt change
            if abs(self.AtmosphericLight_Y - AtmosphericLight_Y) > 50:
                AtmosphericLight_Y = self.AtmosphericLight_Y
                AtmosphericLight = self.AtmosphericLight


    def TransmissionEstimation(self, blk_size):
        maxx = int((self.height // blk_size) * blk_size)
        maxy = int((self.width // blk_size) * blk_size)
        lamdaL = 4
        MinE = np.full(self.imgY.shape, 1e10)
        fOptTrs = np.zeros(self.imgY.shape)
        average = np.zeros(self.imgY.shape)

        for i in range(0, maxx, blk_size):
            for j in range(0, maxy, blk_size):
                average[i:i+blk_size,j:j+blk_size] = self.imgY[i:i+blk_size,j:j+blk_size].mean()

        for t, fTrans in enumerate(np.linspace(0.3,1,8)):
            over255 = np.zeros(self.imgY.shape)
            lower0 = np.zeros(self.imgY.shape)
            transed = (self.imgY.astype(int) - AtmosphericLight_Y)/fTrans + AtmosphericLight_Y

            Econtrast = -(transed - average)**2 / blk_size**2
            over255[transed > 255] = (transed[transed > 255] - 255)**2
            lower0[transed < 0] = (transed[transed < 0])**2
            for i in range(0, maxx, blk_size):
                for j in range(0, maxy, blk_size):
                    E = Econtrast[i:i+blk_size,j:j+blk_size].sum() + lamdaL*(over255[i:i+blk_size,j:j+blk_size].sum() + lower0[i:i+blk_size,j:j+blk_size].sum())
                    if E < MinE[i][j]:
                        MinE[i:i+blk_size,j:j+blk_size] = E
                        fOptTrs[i:i+blk_size,j:j+blk_size] = fTrans
        self.pfTransmission = fOptTrs

    def RestoreImage(self):
        img_out = np.zeros(self.img_input.shape)
        self.pfTransmission = np.maximum(self.pfTransmission, np.full((self.height, self.width), 0.3))
        for i in range(3):
            img_out[:,:,i] = np.clip(((self.img_input[:,:,i].astype(int) - AtmosphericLight[i]) / self.pfTransmission + AtmosphericLight[i]),0,255)

        return img_out

    def GuidedFilter(self, rads, eps):
        ## Kaiming He Guided filtering
        meanI = cv2.boxFilter(self.imgY/255, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        meanP = cv2.boxFilter(self.pfTransmission, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        meanIP = cv2.boxFilter(self.imgY/255*self.pfTransmission, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        covIP = meanIP - meanI * meanP

        meanII = cv2.boxFilter((self.imgY/255)**2, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        varI = meanII - meanI ** 2
        a = covIP / (varI + eps)
        b = meanP - a * meanI
        meanA = cv2.boxFilter(a, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        meanB = cv2.boxFilter(b, -1, (rads,rads), borderType=cv2.BORDER_REPLICATE)
        res = meanA * self.imgY/255 + meanB
        self.pfTransmission = res
        self.pfTransmission = np.maximum(self.pfTransmission, np.full((self.height, self.width), 0.3))  # clip transmission => larger than 0.3
    
def main():
    if args.type == 'image':
        im = cv2.imread(args.input)
        print("Image shape:", im.shape)
        dehaze_img = Dehazing(im)
        dehaze_img.AirLightEstimation((0,0), im.shape[0], im.shape[1])
        blk_size = 16
        dehaze_img.TransmissionEstimation(blk_size)
        eps = 0.001
        dehaze_img.GuidedFilter(20, eps)
        result_img = dehaze_img.RestoreImage().astype('uint8')

        cv2.namedWindow('input_img', cv2.WINDOW_NORMAL)
        cv2.namedWindow('result_img', cv2.WINDOW_NORMAL)
        cv2.imshow('input_img', im)
        cv2.imshow('result_img', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(args.output, result_img)

    elif args.type == 'video':
        video_capture = cv2.VideoCapture(args.input)
        ret, init = video_capture.read()
        h, w = init.shape[0], init.shape[1]
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print("fps:", fps, ", width:", w, ", height:", h)

        video_capture = cv2.VideoCapture(args.input)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        cnt = 0

        while True:
            ret, frame = video_capture.read()
            cv2.namedWindow('input_img', cv2.WINDOW_NORMAL)
            cv2.imshow('input_img', frame)

            if ret == True and cnt % 2 == 0:                   # process every 2 frames -> avoid lag
                dehaze_img = Dehazing(frame)
                if cnt == 0:                                   # use the airlight of the first frame
                    dehaze_img.AirLightEstimation((0,0), frame.shape[0], frame.shape[1])

                blk_size = 8
                dehaze_img.TransmissionEstimation(blk_size)
                eps = 0.001
                dehaze_img.GuidedFilter(20, eps)
                result_img = dehaze_img.RestoreImage().astype('uint8')
                cv2.namedWindow('result_img', cv2.WINDOW_NORMAL)
                cv2.imshow('result_img', result_img)
                out.write(result_img)
                #print(cnt)
            elif ret != True:
                video_capture.release()
                out.release()
                cv2.destroyAllWindows()
                break
            cnt += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):   break

if __name__ == '__main__':
    main()
