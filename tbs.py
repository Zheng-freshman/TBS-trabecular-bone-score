from torchvision import transforms
import cv2
import os
import math
import random
import numpy as np
import torch
from multiprocessing import Pool
from functools import partial
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


tf = transforms.Compose([
    transforms.ToTensor()
])
round = 2*math.pi
k_max = 10


def is_points_in_pic(points, height, width) -> bool:
    """
    :points [[x1,y1][x2,y2]...]
    :x which line, height of pixel
    :y which row, width of pixel
    """
    result = []
    for p in points:
        if p[0]>0 and p[0]<height and p[1]>0 and p[1]<width:
            result.append(p)
    return np.array(result)


def is_points_in_mask(points, mask) -> bool:
    """
    :points [[x1,y1][x2,y2]...]
    :x which line, height of pixel
    :y which row, width of pixel
    """
    result = []
    for p in points:
        if mask[p[0]][p[1]] != 0:
            result.append(p)
    return np.array(result)


def screen(points, height, width, mask=None): #过滤超出范围的点
    if mask is not None:
        points = is_points_in_mask(points, mask)
    points = is_points_in_pic(points, height, width)
    #print(len(points))
    return points



def tbsVar(gray_img, mask, i):
    dir_num=5 #取多个方向
    sub_round = round/dir_num
    height = gray_img.size(0)
    width = gray_img.size(1)
    V_line = torch.zeros(k_max, width) #每个像素每个k值下的结果
    sub_pixels = torch.zeros(k_max) #距离为k时，该行有几个点没算方差
    for j in range(0, width):
        if(mask[i][j]==0):
            for k in range(0, k_max):
                V_line[k][j] = 0
                sub_pixels[k] += 1
            continue
        init = random.random()*round
        dir_np = np.array([init+sub_round*dir for dir in range(0, dir_num)])
        dx_np = np.sin(dir_np)
        dy_np = np.cos(dir_np)
        for k in range(0, k_max):
            i2 = np.floor(i+0.5+(k+1)*dx_np).astype(np.int32)
            j2 = np.floor(j+0.5+(k+1)*dy_np).astype(np.int32)
            points = np.concatenate((np.expand_dims(i2,0),np.expand_dims(j2,0))).transpose()
            points_flited = screen(points, height, width, mask)
            if len(points_flited) == 0:
                V_line[k][j] = 0
                sub_pixels[k] += 1
            else:
                V_line_list = []
                for p in points_flited:
                    temp = (gray_img[i][j]-gray_img[p[0]][p[1]])**2
                    V_line_list.append(temp)
                V_line[k][j] = sum(V_line_list)/len(V_line_list)
    #print("line {0} done".format(i))
    #with open("output2.txt", 'a') as f: print(sub_pixels, file=f)
    return i, V_line, sub_pixels


def Fun(p, x):
    a1,a2,a3 = p
    return a1*x**2+a2*x+a3
def dFun(p, x):
    a1,a2,a3 = p
    return 2*a1*x+a2
def error(p,x,y):
    return Fun(p,x)-y

def tbs_per(image, savepath, savename, mask=None):
    #img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    #cv2.imshow("11",gray_img)
    #cv2.waitKey()
    tensor_img = tf(image)
    gray_img = tensor_img[0]
    height = gray_img.size(0)
    width = gray_img.size(1)
    print("img ready")
    x_axis = np.array([math.log10(k+1) for k in range(0, k_max)])

    V_lines = []
    V_img = torch.zeros(k_max, height, width)
    #V_img = np.zeros((k_max, height, width))
    sum_pixels = []
    for k in range(0, k_max):
        sum_pixels.append(height*width)
    sum_pixels = torch.tensor(sum_pixels)
    print("readying pool")

    with Pool() as p: #多线程，每线程一行
        tbs_reduce = p.map(partial(tbsVar, gray_img, mask), range(0, height)) #可能是乱序
    """
    tbs_reduce = []
    for i in range(0, height):
        temp = tbsVar(gray_img, mask, i)
        tbs_reduce.append(temp)
    """
    #print(angel_img)
    tbs_reduce.sort(key=lambda line: line[0]) #h*[k,w]
    for line in tbs_reduce:
        sum_pixels = sum_pixels - line[2]
    V_img = [line[1].unsqueeze(0) for line in tbs_reduce] #h*[1, k, w]
    V_img = torch.cat(V_img) #h,k,w
    V_img = V_img.transpose(1,0) #k,h,w
    V_img = V_img.reshape(k_max, -1) #k,h*w
    V_sum = torch.sum(V_img, dim=1)
    V_mean = V_sum / sum_pixels
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].plot(np.array([(k+1) for k in range(0, k_max)]), V_mean, 'r', label = 'orgin')
    axs[0].legend()

    #print(V_mean)
    V_log = torch.log1p(V_mean)
    y_axis = V_log / V_log.tolist()[0]
    y_axis = y_axis.numpy()
    #print(x_axis)
    #print(y_axis)

    print("fitting")
    p0 = [0.1, -0.01, 100]
    para = leastsq(error, p0, args=(x_axis, y_axis))
    axs[1].plot(x_axis, y_axis, 'r', label = 'orgin')
    y_fitted = Fun(para[0], x_axis)
    axs[1].plot(x_axis, y_fitted, 'b', label = 'fitted')
    axs[1].legend()
    #plt.show()
    plt.savefig(os.path.join(savepath,savename), bbox_inches='tight')
    tbs_value = dFun(para[0], 0)
    #print(para[0])
    print(filename+": "+str(tbs_value))


if __name__ == "__main__":
    path="data"
    imgpath = "data/GRAY"
    savepath = "data/result"
    maskfile = "data/coco.json"
    #filename = "test.jpg"
    #savename = "test.png"
    coco = COCO(maskfile)
    imgIds = coco.getImgIds()
    for imgId in imgIds:
        ann = coco.loadAnns([imgId])[0]
        seg = ann["segmentation"]
        mask = coco.annToMask(ann)
        #np.set_printoptions(threshold=np.inf)
        #with open('output.txt', 'w') as f: print(mask, file=f)
        #show_numpy(np.array([mask]))
        imginfo = coco.loadImgs([imgId])[0]
        filename = imginfo["file_name"]
        savename = filename[:-4]+".png"
        img_file = cv2.imread(os.path.join(imgpath,filename))
        img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
        tbs_per(img, savepath, savename, mask)
