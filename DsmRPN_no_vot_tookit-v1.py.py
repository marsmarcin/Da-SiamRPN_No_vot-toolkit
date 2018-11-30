import cv2
from net import SiamRPNBIG
import torch
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
import numpy as np 
import os
target_path='/home/mars/Documents/Subject2018/vot2018-SiamRPN/DaSiamRPN-master/code/sequences/'

sequences_name=target_path+'ants1/'

image_path = sequences_name+'color/'

ground_truth_path = sequences_name+'/groundtruth.txt'
list0=[]
for files in os.walk(image_path):  
    list0.append(files)
# print(list0[0][2])
img_list=[]
for i1 in list0[0][2]:
    img_list.append(i1)
img_name=np.array(img_list)
img_name.sort()
#print(img_name)
data=[]
for i2 in open(ground_truth_path):
    data.append(i2)
data1=[]
for i in data:
    data1.append(i[:len(i)-1])
data2=[]
data3=[]
for j in data1:
    data2.append(j.split(','))
# print(data2[1])
data4=[]
for k in data2:
    data3=np.array([0.0 if y=='' else float(y) for y in k])
    data4.append(data3)
data_rect=np.array(data4)
# print(data_rect[0][0])
x1=data_rect[0][0]
y1=data_rect[0][1]
x2=data_rect[0][2]
y2=data_rect[0][3]
x3=data_rect[0][4]
y3=data_rect[0][5]
x4=data_rect[0][6]
y4=data_rect[0][7]
net = SiamRPNBIG()
net.load_state_dict(torch.load('/home/mars/Documents/Subject2018/vot2018-SiamRPN/DaSiamRPN-master/code/SiamRPNBIG.model'))
net.eval().cuda()

for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

cx = (x1+x2+x3+x4)/4
cy = (y1+y2+y3+y4)/4
w = ((x2-x1)+(x3-x4))/2
h = ((y3-y2)+(y4-y1))/2

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

image_file =image_path + img_name[0]

im = cv2.imread(image_file)  # HxWxC

# cv2.imshow("hh",im)

# cv2.waitKey(0)

state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker

state = SiamRPN_track(state, im)  # track

res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

for ii in range(1,len(img_name)):
    cx = int(res[0]+res[2]/2)
    cy = int(res[1]+res[3]/2)
    w = int(res[2])
    h = int(res[3])
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    
    im_temp = cv2.imread(image_path+img_name[ii])
    state = SiamRPN_track(state, im_temp)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    
    # print(ii)
    # print(img_name[ii])
    # print(res)
    xx = int(res[0])
    yy = int(res[1])
    ww = int(res[2])
    hh = int(res[3])

    cv2.rectangle(im_temp,(xx,yy),(xx+ww,yy+hh),(0,0,255))
    cv2.imshow("prediction",im_temp)
    cv2.waitKey(0)
