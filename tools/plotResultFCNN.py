#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import re
import pdb
import sys
def getLoss(file):
    f = open(file, 'r')
    iter = []
    total_loss = []
    rpn_loss_cls = []
    rpn_loss_box = []
    loss_cls = []
    loss_box = []
    count = 1
    totalcount = 1
    for line in f:
       # pdb.set_trace()
        if (re.search(r'loss_cls: [0-9].[0-9]*', line) != None):
            #pdb.set_trace()
            ### find total loss
            try:
                str_temp = re.findall(r'total loss: [0-9].[0-9]*,', line)[0]
                val = float(re.findall(r'[0-9].[0-9]*', str_temp)[0])
            except:
                val = total_loss[-1]
            total_loss.append(val)

            ### find rpn_loss_cls
            try:
                str_temp = re.findall(r'rpn_loss_cls: [0-9].[0-9]*,', line)[0]
                val = float(re.findall(r'[0-9].[0-9]*', str_temp)[0])
            except:
                val = rpn_loss_cls[-1]
            rpn_loss_cls.append(val)

            ### find rpn_loss_box
            try:
                str_temp = re.findall(r'rpn_loss_box: [0-9].[0-9]*,', line)[0]
                val = float(re.findall(r'[0-9].[0-9]*', str_temp)[0])
            except:
                val = rpn_loss_cls[-1]
            rpn_loss_box.append(val)

            ### find loss_cls
            try:
                str_temp = re.findall(r' loss_cls: [0-9].[0-9]*,', line)[0]
                val = float(re.findall(r'[0-9].[0-9]*', str_temp)[0])
            except:
                val = loss_cls[-1]
            loss_cls.append(val)

            ### find loss_cls
            try:
                str_temp = re.findall(r' loss_box: [0-9].[0-9]*,', line)[0]
                val = float(re.findall(r'[0-9].[0-9]*', str_temp)[0])
            except:
                val = loss_box[-1]
            loss_box.append(val)

            ### iter
            iter.append(count)
            count += 1
        totalcount += 1
    print ('total line ' + str(totalcount))
    return iter, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box



if __name__ == '__main__':
    #str_file = 'train_driving.out'
    str_file = sys.argv[1]
    iter, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box = getLoss(str_file)

    numiter = len(iter)
    number = numiter / 10

    iter1 = []
    total_loss1 = []
    rpn_loss_cls1 = []
    rpn_loss_box1 = []
    loss_cls1 = []
    loss_box1 = []
    for i in range(number):
        temp_total = 0
        temp_rpn_cls = 0
        temp_rpn_box = 0
        temp_cls = 0
        temp_box = 0
        for j in range(10):
            temp_total += total_loss[i * 10 + j]
            temp_rpn_cls += rpn_loss_cls[i * 10 + j]
            temp_rpn_box += rpn_loss_box[i * 10 + j]
            temp_cls += loss_cls[i * 10 + j]
            temp_box += loss_box[i * 10 + j]
            if (j == 0): iter1.append(i * 10 + j)
        total_loss1.append(temp_total / 10)
        rpn_loss_cls1.append(temp_rpn_cls / 10)
        rpn_loss_box1.append(temp_rpn_box / 10)
        loss_cls1.append(temp_cls / 10)
        loss_box1.append(temp_box / 10)

   # pdb.set_trace()
    iter_np = np.asarray(iter1)
    total_loss_np = np.asarray(total_loss1)
    rpn_loss_cls_np = np.asarray(rpn_loss_cls1)
    rpn_loss_box_np = np.asarray(rpn_loss_box1)
    loss_cls_np = np.asarray(loss_cls1)
    loss_box_np = np.asarray(loss_box1)
   # pdb.set_trace()

    plt.figure(1)
    plt.plot(iter_np, total_loss_np)
    plt.title("total loss")
    plt.savefig('total_loss.png')

    plt.figure(2)
    plt.plot(iter_np, rpn_loss_cls_np)
    plt.title("rpn_loss_cls")
    plt.savefig('rpn_loss_cls.png')

    plt.figure(3)
    plt.plot(iter_np, rpn_loss_box_np)
    plt.title("rpn_loss_box")
    plt.savefig('rpn_loss_box.png')

    plt.figure(4)
    plt.plot(iter_np, loss_cls_np)
    plt.title("loss_cls")
    plt.savefig('loss_cls.png')

    plt.figure(5)
    plt.plot(iter_np, loss_box_np)
    plt.title("loss_box")
    plt.savefig('loss_box.png')
    plt.show()
