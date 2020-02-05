# Differentiable plasticity: natural image memorization and reconstruction.
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


# This program uses the click module rather than argparse to scan command-line arguments. I won't do that again.

# You start getting acceptable results after ~3000 episodes (~15 minutes with a standard GPU). Let it run longer for better results.

# To observe the results, run testpics.py (which uses the output files produced by this program)


import torch
import torch.nn as nn
from torch.autograd import Variable
import click
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

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorboardX import SummaryWriter

# Uber-only:
# import OpusHdfsCopy
# from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs


# Loading the image data. This requires downloading the CIFAR 10 dataset (Python version) - https://www.cs.toronto.edu/~kriz/cifar.html

###########       loading track data      ############
IMGSIZE = 32
OTBDIR = '../testing_dataset/OTB100/'
datafileList = os.listdir(OTBDIR)
datafileList.sort()
img = []
# for numfile in range(len(datafileList)):
for numfile in range(1):
    OTBSUBDIR = OTBDIR + datafileList[3]  # [numfile]  dataset Basketball
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
    print("img:")
    print(img, img.shape)

############################################################

np.set_printoptions(precision=4)

defaultParams = {
    'nbpatterns': 2,  # number of images per episode
    'nbprescycles': 3,  # number of presentations for each image
    'prestime': 20,  # number of time steps for each image presentation
    'prestimetest': 3,  # number of time steps for the test (degraded) image
    'interpresdelay': 2,  # number of time steps (with zero input) between two presentations
    'patternsize': 1024,  # size of the images (32 x 32 = 1024)
    'nbiter': 100000,  # number of episodes
    'probadegrade': .5,
    # when contiguousperturbation is False (which it shouldn't be), probability of zeroing each pixel in the test image
    'lr': 1e-4,  # Adam learning rate
    'print_every': 10,  # how often to print statistics and save files
    'homogenous': 0,  # whether alpha should be shared across connections
    'rngseed': 0  # random seed
}

# ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;  # For GPU


# Generate GT0 TI TI_ GTi+1
def generateGTArr(params, i, yout, Tout):  # ,Ti_,ti
    inputT = np.zeros((params['nbsteps'], 1, params['nbneur']))
    patterns = []
    GT0 = img[0].reshape((3, 1024)).sum(0).astype(float)
    GT0 = GT0[:params['patternsize']]
    GT0 = GT0 - np.mean(GT0)
    GT0 = GT0 / (1e-8 + np.max(np.abs(GT0)))
    patterns.append(GT0)
    # print("GT0")
    # print(GT0)
    # patterns.append(Ti)
    Ti = img[i].reshape((3, 1024)).sum(0).astype(float)
    Ti = Ti[:params['patternsize']]
    Ti = Ti - np.mean(Ti)
    Ti = Ti / (1e-8 + np.max(np.abs(Ti)))
    # print("Ti")
    # print(Ti)
    patterns.append(Ti)

    if Tout == False:
        _Ti = Ti
        patterns.append(_Ti)  # use ti as ti_
    else:
        _Ti = yout  # output
        patterns.append(_Ti)
    # print("_Ti")
    # print(_Ti)
    # print("pattern:")
    # print(patterns)

    GTi1 = img[i + 1].reshape((3, 1024)).sum(0).astype(float)
    GTi1 = GTi1[:params['patternsize']]
    GTi1 = GTi1 - np.mean(GTi1)
    GTi1 = GTi1 / (1e-8 + np.max(np.abs(GTi1)))
    targetpattern = GTi1.copy()

    # print("targetpattern:")
    # print(targetpattern)

    # preservedbits = np.ones(params['patternsize'])
    # Inserting the inputs in the input tensor at the proper places
    for nc in range(params['nbprescycles']):  # 3
        np.random.shuffle(patterns)
        for ii in range(params['nbpatterns']):  # 3
            for nn in range(params['prestime']):  # 20
                numi = nc * (params['nbpatterns'] * (params['prestime'])) + ii * (
                    params['prestime']) + nn
                # print("numi:")
                # print(numi)
                inputT[numi][0][:params['patternsize']] = patterns[ii][:]

    for nn in range(params['nbsteps']):
        inputT[nn][0][-1] = 1.0  # Bias neuron is forced to 1
        # inputT[nn] *= params['inputboost']       # Strengthen inputs

    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    target = torch.from_numpy(targetpattern).type(ttype)

    return inputT, target


##################################


def generateTrainArr(params, yout, Tout):  # ,Ti_,ti
    inputT = np.zeros((params['nbsteps'], 1, params['nbneur']))
    patterns = []
    GT0 = img[0].reshape((3, 1024)).sum(0).astype(float)
    GT0 = GT0[:params['patternsize']]
    GT0 = GT0 - np.mean(GT0)
    GT0 = GT0 / (1e-8 + np.max(np.abs(GT0)))
    patterns.append(GT0)
    # print("GT0")
    # print(GT0)
    # patterns.append(Ti)
    i = np.random.randint(img.shape[0] - 1)
    Ti = img[i].reshape((3, 1024)).sum(0).astype(float)
    Ti = Ti[:params['patternsize']]
    Ti = Ti - np.mean(Ti)
    Ti = Ti / (1e-8 + np.max(np.abs(Ti)))
    # print("Ti")
    # print(Ti)
    patterns.append(Ti)

    if Tout == False:
        _Ti = Ti
        patterns.append(_Ti)  # use ti as ti_
    else:
        _Ti = yout  # output
        patterns.append(_Ti)
    # print("_Ti")
    # print(_Ti)
    # print("pattern:")
    # print(patterns)

    GTi1 = img[i + 1].reshape((3, 1024)).sum(0).astype(float)
    GTi1 = GTi1[:params['patternsize']]
    GTi1 = GTi1 - np.mean(GTi1)
    GTi1 = GTi1 / (1e-8 + np.max(np.abs(GTi1)))
    targetpattern = GTi1.copy()

    # print("targetpattern:")
    # print(targetpattern)

    # preservedbits = np.ones(params['patternsize'])
    # Inserting the inputs in the input tensor at the proper places
    for nc in range(params['nbprescycles']):  # 3
        # np.random.shuffle(patterns)

        for ii in range(params['nbpatterns']):  # 3
            for nn in range(params['prestime']):  # 20
                numi = nc * (params['nbpatterns'] * (params['prestime'])) + ii * (
                    params['prestime']) + nn
                # print("numi:",numi)
                inputT[numi][0][:params['patternsize']] = patterns[ii][:]

    for nn in range(params['nbsteps']):
        inputT[nn][0][-1] = 1.0  # Bias neuron is forced to 1
        # inputT[nn] *= params['inputboost']       # Strengthen inputs

    inputT[0][0][:params['patternsize']] = patterns[0][:]
    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    target = torch.from_numpy(targetpattern).type(ttype)

    return inputT, target

class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the comp neuro order, following deep learning / pytorch conventions
        # Each *column* of w targets a single output neuron
        self.w = Variable(.01 * torch.randn(params['nbneur'], params['nbneur']).type(ttype),
                          requires_grad=True)  # fixed (baseline) weights
        if params['homogenous'] == 1:
            self.alpha = Variable(.01 * torch.ones(1).type(ttype),
                                  requires_grad=True)  # plasticity coefficients: homogenous/shared across connections
        else:
            self.alpha = Variable(.01 * torch.randn(params['nbneur'], params['nbneur']).type(ttype),
                                  requires_grad=True)  # plasticity coefficients: independent
        self.eta = Variable(.01 * torch.ones(1).type(ttype),
                            requires_grad=True)  # "learning rate" of plasticity, shared across all connections
        self.params = params

    def forward(self, input, yin, hebb):
        # Inputs are fed by clamping the output of cells that receive input at the input value, like in standard Hopfield networks
        # clamps = torch.zeros(1, self.params['nbneur'])
        clamps = np.zeros(self.params['nbneur'])
        # print("CLAMPS 1 ")
        # print(clamps,clamps.shape)

        zz = torch.nonzero(input.data[0].cpu()).numpy().squeeze()
        # print("ZZ ZZ ZZ ZZ ZZ ZZ ")
        # print(zz, zz.shape)
        clamps[zz] = 1
        # print(clamps)
        # print(1-clamps)
        # print(input)
        # print(yin)
        clamps = Variable(torch.from_numpy(clamps).type(ttype), requires_grad=False).float()

        # writer.add_image('{input}_feature_maps', input, global_step=0)
        # print("clamps:")
        # print(clamps)
        # print(1-clamps)
        yout = F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb))) * (1 - clamps) + input * clamps
        hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0]  # bmm used to implement outer product

        # print("YIN YOUT ")
        # print(yin.shape)
        # print(yout.shape)
        # print(yin)
        # print(self.w,self.w.shape)
        # print(self.alpha,self.alpha.shape)
        # print(F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb))))
        # print(input,input.shape)
        # print(yout, yout.shape)
        # print("Hebb:")
        # print(hebb)
        return yout, hebb

    def initialZeroState(self):
        return Variable(torch.zeros(1, self.params['nbneur']).type(ttype))

    def initialZeroHebb(self):
        return Variable(torch.zeros(self.params['nbneur'], self.params['nbneur']).type(ttype))


def train(paramdict=None):
    # params = dict(click.get_current_context().params)
    print("Starting training...")
    params = {}
    params.update(defaultParams)
    if paramdict:
        params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    sys.stdout.flush()
    params['nbsteps'] = params['nbprescycles'] * (
                (params['prestime'] + params['interpresdelay']) * params['nbpatterns']) + params[
                            'prestimetest']  # Total number of steps per episode
    params['nbneur'] = params['patternsize'] + 1
    suffix = "images_" + "".join([str(x) + "_" if pair[0] is not 'nbneur' and pair[0] is not 'nbsteps' and pair[
        0] is not 'print_every' and pair[0] is not 'rngseed' else '' for pair in zip(params.keys(), params.values()) for
                                  x in pair])[:-1] + '_rngseed_' + str(
        params['rngseed'])  # Turning the parameters into a nice suffix for filenames; rngseed always appears last

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rngseed']);
    random.seed(params['rngseed']);
    torch.manual_seed(params['rngseed'])
    # print(click.get_current_context().params)

    print("Initializing network")
    net = Network(params)
    total_loss = 0.0

    print("Initializing optimizer")
    optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=params['lr'])
    all_losses = []
    # print_every = 20
    nowtime = time.time()
    print("Starting episodes...")
    sys.stdout.flush()

    yout=[]
    for numiter in range(params['nbiter']):
        # print("Iter ", numiter)
        # sys.stdout.flush()
        y = net.initialZeroState()
        hebb = net.initialZeroHebb()
        optimizer.zero_grad()

        # inputs, target = generateTestInput(params)
        inputs, target = generateTrainArr(params, y, Tout=False)

        # Running the episode
        for numstep in range(params['nbsteps']):
            y, hebb = net(Variable(inputs[numstep], requires_grad=False), y, hebb)

        # Computing gradients, applying optimizer
        loss = (y[0][:params['patternsize']] - Variable(target, requires_grad=False)).pow(2).sum()
        loss.backward()
        optimizer.step()

        yout = y

        # lossnum = loss.data[0]
        lossnum = loss.item()
        total_loss += lossnum

        # Printing statistics, saving files
        if (numiter + 1) % params['print_every'] == 0:

            print(numiter, "====")
            td = target.cpu().numpy()
            yd = y.data.cpu().numpy()[0][:-1]
            print("y: ", yd[:10])
            print("target: ", td[:10])

            absdiff = np.abs(td - yd)
            print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
            print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])

            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['print_every'], "iters: ", nowtime - previoustime)
            total_loss /= params['print_every']
            all_losses.append(total_loss)
            print("Mean loss over last", params['print_every'], "iters:", total_loss)
            print("Saving local files...")
            sys.stdout.flush()
            with open('results_' + suffix + '.dat', 'wb') as fo:
                pickle.dump(net.w.data.cpu().numpy(), fo)
                pickle.dump(net.alpha.data.cpu().numpy(), fo)
                pickle.dump(net.eta.data.cpu().numpy(), fo)
                pickle.dump(all_losses, fo)
                pickle.dump(params, fo)
            print("ETA:", net.eta.data.cpu().numpy())
            with open('loss_' + suffix + '.txt', 'w') as thefile:
                for item in all_losses:
                    thefile.write("%s\n" % item)

            sys.stdout.flush()
            sys.stderr.flush()

            total_loss = 0
        print("yout:",yout)

        for nc in range(params['nbprescycles']):  # 3
        # np.random.shuffle(patterns)
            for nn in range(params['prestime']):  # 20
                numi = nc * (params['nbpatterns'] * (params['prestime'])) + 2 * (  #2 ii
                        params['prestime']) + nn
                inputs[numi][0][:params['patternsize']] = yout



@click.command()
@click.option('--nbpatterns', default=defaultParams['nbpatterns'])
@click.option('--nbprescycles', default=defaultParams['nbprescycles'])
@click.option('--homogenous', default=defaultParams['prestime'])
@click.option('--prestime', default=defaultParams['prestime'])
@click.option('--prestimetest', default=defaultParams['prestimetest'])
@click.option('--interpresdelay', default=defaultParams['interpresdelay'])
@click.option('--patternsize', default=defaultParams['patternsize'])
@click.option('--nbiter', default=defaultParams['nbiter'])
@click.option('--probadegrade', default=defaultParams['probadegrade'])
@click.option('--lr', default=defaultParams['lr'])
@click.option('--print_every', default=defaultParams['print_every'])
@click.option('--rngseed', default=defaultParams['rngseed'])
def main(nbpatterns, nbprescycles, homogenous, prestime, prestimetest, interpresdelay, patternsize, nbiter,
         probadegrade, lr, print_every, rngseed):
    train(paramdict=dict(click.get_current_context().params))
    # print(dict(click.get_current_context().params))

    # writer = SummaryWriter('./Result')
    model_m = Network(params)
    # dum_input = np.zeros((1, params['nbneur']))
    dum_imput = Variable(inputs[numstep], requires_grad=False)

    dum_in = model_m.initialZeroState()
    dum_hebb = net.initialZeroHebb()
    dum_y, dum_hebb = net(dum_input, dum_y, dum_hebb)

    with SummaryWriter(comment='PlasNetwork') as W:
        W.add_graph(net, (dum_in, y, hebb), verbose=True)

    with SummaryWriter(comment='PlasNetwork') as W:
        W.add_graph(model_m, dum_input, dum_in, dum_hebb)


if __name__ == "__main__":
    # train()
    main()

