# Generate a figure that shows a number of episodes
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.




import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
import random
import sys
import pickle
import pdb
import time
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
plt.ion()


#import plastic_train as pics
from plastic_train import Network


ttype = torch.cuda.FloatTensor

def loadImg(videoNum):
    IMGSIZE = 32
    OTBDIR = '../testing_dataset/OTB100/'
    datafileList = os.listdir(OTBDIR)
    datafileList.sort()
    img = []
    # for numfile in range(len(datafileList)):
    for numfile in range(1):
        OTBSUBDIR = OTBDIR + datafileList[videoNum]  # [numfile]  dataset Basketball
        rectFile = OTBSUBDIR + '/groundtruth_rect.txt'
        rectArr = []
        with open(rectFile, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                # print(lines,lines.type)
                if not lines:
                    break
                    pass
                p_tmp = [int(i) for i in lines.split(',')]
                rectArr.append(p_tmp)
            rectArr = np.array(rectArr)
            pass

        # print(rectArr)
        # print(rectArr[1])

        imgfileList = os.listdir(OTBSUBDIR + '/img')
        imgfileList.sort()

        # print(imgfileList)
        # print(len(imgfileList))

        for numfile in range(len(imgfileList)):
            picFile = OTBSUBDIR + '/img/' + imgfileList[numfile]
            pic = mpimg.imread(picFile)
            # plt.imshow(pic)
            # plt.show()
            rectImg = rectArr[numfile]
            picRect = pic[rectImg[1]:rectImg[1] + rectImg[3], rectImg[0]:rectImg[0] + rectImg[2]]
            # plt.imshow(picRect)
            # plt.show()
            picRectResize = cv2.resize(picRect, (IMGSIZE, IMGSIZE))
            # plt.imshow(picRectResize)
            # plt.show()
            # print(picRectResize,picRectResize.shape)
            # picRect2d = np.concatenate(picRect)
            picRect2d = np.concatenate(picRectResize)
            picRect1d = np.concatenate(picRect2d)
            img.append(picRect1d)
        img = np.array(img)
    return img


def generateInputs(params,GT0,Ti):
    inputT = np.zeros((params['nbsteps'], 1, params['nbneur']))
    patterns = []
    patterns.append(GT0)
    patterns.append(Ti)

    for nc in range(params['nbprescycles']):  # 3
        np.random.shuffle(patterns)
        for ii in range(2):  # 2 #params['nbpatterns']
            for nn in range(params['prestime']):  # 20
                numi = nc * (params['nbpatterns'] * (params['prestime'])) + ii * (
                    params['prestime']) + nn
                inputT[numi][0][:params['patternsize']] = patterns[ii][:]

    for nn in range(params['nbsteps']):
        inputT[nn][0][-1] = 1.0  # Bias neuron is forced to 1

    ttype = torch.cuda.FloatTensor;
    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor


    return inputT

def PlasticNet(GT0,zf,IS_TI,_zf=0):
    suffix = 'images_nbpatterns_2_nbprescycles_3_prestime_20_prestimetest_3_interpresdelay_2_patternsize_1024_nbiter_100000_probadegrade_0.5_lr_0.0001_homogenous_20_rngseed_0'
    fn = './results_' + suffix + '.dat'

    with open(fn, 'rb') as fo:
        myw = pickle.load(fo)
        myalpha = pickle.load(fo)
        myeta = pickle.load(fo)
        myall_losses = pickle.load(fo)
        myparams = pickle.load(fo)

    net = Network(myparams)

    rngseed = 4
    np.random.seed(rngseed);
    random.seed(rngseed);
    torch.manual_seed(rngseed)

    #ttype = torch.cuda.FloatTensor  # Must match the one in pics_eta.py
    # ttype = torch.FloatTensor # Must match the one in pics_eta.py
    net.w.data = torch.from_numpy(myw).type(ttype)
    net.alpha.data = torch.from_numpy(myalpha).type(ttype)
    net.eta.data = torch.from_numpy(myeta).type(ttype)
    ####################################################################
    C = zf[0].shape[1]
    W = zf[0].shape[2]
    H = zf[0].shape[3]
    zf0 = zf[0]
    zf1 = zf[1]
    zf2 = zf[2]
    zfz = torch.cat([zf0, zf1, zf2], 0)
    zfzz = zfz.cpu()
    zfz4d = zfzz.data.numpy()
    zfz3d = np.concatenate(zfz4d)
    zfz2d = np.concatenate(zfz3d)
    zfz1d = np.concatenate(zfz2d)
    Ti = zfz1d


    #############################################################
    if IS_TI:
        _C = _zf[0].shape[1]
        _W = _zf[0].shape[2]
        _H = _zf[0].shape[3]
        zf0 = _zf[0]
        zf1 = _zf[1]
        zf2 = _zf[2]
        zfz = torch.cat([zf0, zf1, zf2], 0)
        zfzz = zfz.cpu()
        zfz4d = zfzz.data.numpy()
        zfz3d = np.concatenate(zfz4d)
        zfz2d = np.concatenate(zfz3d)
        zfz1d = np.concatenate(zfz2d)
        _Ti = zfz1d
        y = _Ti

    else:
        y = net.initialZeroState()


    hebb = net.initialZeroHebb()
    inputsTensor = generateInputs(myparams, GT0, Ti)

    for numstep in range(myparams['nbsteps']):
        y, hebb = net(Variable(inputsTensor[numstep], requires_grad=False), y, hebb)
    zfzdd = y
    #####################################################
    re_zfz4d = zfzdd.reshape(3, C, W, H)
    re_zfzz = torch.from_numpy(re_zfz4d)
    # absdiff=np.abs(zfz4d-re_zfz4d)
    re_zf = []
    re_zf.append(re_zfzz[0])
    re_zf.append(re_zfzz[1])
    re_zf.append(re_zfzz[2])
    _Ti = re_zf
    #####################################################
    return _Ti


def main():
    PATTERNSIZE = 1024
    videoNum = 10
    img = loadImg(videoNum)

    print(img.shape[0])


    GT0 = img[0].reshape((3, 1024)).sum(0).astype(float)
    GT0 = GT0[:PATTERNSIZE]
    GT0 = GT0 - np.mean(GT0)
    GT0 = GT0 / (1e-8 + np.max(np.abs(GT0)))

    Ti = img[1].reshape((3, 1024)).sum(0).astype(float)
    Ti = Ti[:PATTERNSIZE]
    Ti = Ti - np.mean(Ti)
    Ti = Ti / (1e-8 + np.max(np.abs(Ti)))
    IF_TI = False
    _Ti = PlasticNet(GT0, Ti, IF_TI)

    for frameNum in range(1): #img.shape[0]-1
        #print(frameNum)
        Ti = img[frameNum+1].reshape((3, 1024)).sum(0).astype(float)
        Ti = Ti[:PATTERNSIZE]
        Ti = Ti - np.mean(Ti)
        Ti = Ti / (1e-8 + np.max(np.abs(Ti)))

        GTi1 = img[frameNum + 2].reshape((3, 1024)).sum(0).astype(float)
        GTi1 = GTi1[:PATTERNSIZE]
        GTi1 = GTi1 - np.mean(GTi1)
        GTi1 = GTi1 / (1e-8 + np.max(np.abs(GTi1)))
        targetpattern = GTi1.copy()
        targetpattern=torch.from_numpy(targetpattern).type(ttype)

        IF_TI = True
        _Ti = PlasticNet(GT0,Ti,IF_TI,_Ti)

    print("_Ti:",_Ti)


    td = targetpattern.cpu().numpy()
    yd = _Ti.data.cpu().numpy()[0][:-1]
    absdiff = np.abs(td - yd)
    print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
    print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])

    imagesize = int(np.sqrt(PATTERNSIZE))

    GT0 = torch.from_numpy(GT0).type(ttype)
    Ti = torch.from_numpy(Ti).type(ttype)

    gt0 = GT0.cpu().numpy().reshape((imagesize, imagesize))
    ti = Ti.cpu().numpy().reshape((imagesize, imagesize))

    _ti = _Ti.data.cpu().numpy()[0][:-1].reshape((imagesize, imagesize))
    gti1 = targetpattern.cpu().numpy().reshape((imagesize, imagesize))

    nn = 1
    plt.subplot(1, 5, nn)
    plt.axis('off')
    plt.imshow(gt0, cmap='gray', vmin=-1.0, vmax=1.0)

    plt.subplot(1, 5, nn+1)
    plt.axis('off')
    plt.imshow(ti, cmap='gray', vmin=-1.0, vmax=1.0)

    plt.subplot(1, 5, nn+2)
    plt.axis('off')
    plt.imshow(_ti, cmap='gray', vmin=-1.0, vmax=1.0)

    plt.subplot(1, 5, nn+3)
    plt.axis('off')
    plt.imshow(gti1, cmap='gray', vmin=-1.0, vmax=1.0)

    plt.show()




if __name__ == "__main__":
    # train()
    main()

