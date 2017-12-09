import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import pdb as pdb
namepath = '../data/kitti_train.txt'
filepath = '../data/training/image_2/';
imgpath = './results/img1_out/'
txtpath = './results/'
CLASSES = ('__background__','car')
CLASSES2 = ('__background__', 'Car')
       #    'aeroplane', 'bicycle', 'bird', 'boat',
        #   'bottle', 'bus', 'car', 'cat', 'chair',
         #  'cow', 'diningtable', 'dog', 'horse',
          # 'motorbike', 'person', 'pottedplant',
          # 'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')
def demo2(sess, net, image_name, CONF_THRESH, NMS_THRESH):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = filepath + image_name
    #pdb.set_trace()
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

  #  CONF_THRESH = 0.8
  #  NMS_THRESH = 0.3
    res = np.array([])
   # pdb.set_trace()
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
       # cls_group = np.ones(boxes.shape[0]) * cls_ind;
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
       # dets = np.hstack((cls_boxes, cls_group[:, np.newaxis],
        #                  cls_scores[:, np.newaxis])).astype(np.float32)
        #pdb.set_trace()
	keep = nms(dets, NMS_THRESH)
        size_temp = len(keep)
        cls_group = np.ones(size_temp) * cls_ind;
        dets = dets[keep, :]
        dets = np.hstack((dets, cls_group[:, np.newaxis])).astype(np.float32)
        inds = np.where(dets[:, -2] >= CONF_THRESH)[0]
        #print "length inds", len(inds)
        dets = dets[inds, :]
        if (cls_ind == 1):
            res = dets
        else:
            #pdb.set_trace()
            res = np.vstack((res, dets))
    return res;


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    nms1 = [0.2, 0.3, 0.4, 0.5]
    conf = [0.6, 0.7, 0.8, 0.9]
    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session

    config=tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    f = open(namepath, 'r')
    testFiles = []
    for line in f:
	name = line.strip('\r\n')
	testFiles.append(name)
    print testFiles[0:10]
    numFile = len(testFiles)
 #   numFile = 2   
    for k in range(len(conf)):
	for j in range(len(nms1)):
	    folder = 'c_' + str(conf[k]) + '_n_' + str(nms1[j]) + '/'
	    folder_name = txtpath + folder 
	    if (not os.path.exists(folder_name)):
		os.mkdir(folder_name)
	    for i in range(numFile):
		im_name = testFiles[i] + '.png'
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for {}'.format(im_name)
		res = demo2(sess, net, im_name, conf[k], nms1[j])
		numBox = res.shape[0]
		tofile_txt = folder_name + testFiles[i]+ '.txt'
		ftxt = open(tofile_txt,'w')
		for i in range(numBox):
		    ftxt.write(CLASSES2[int(res[i][5])] + ' ' + str(res[i][0]) + ' ' + str(res[i][1]) + ' ' + str(res[i][2]) + ' ' + str(res[i][3]) + ' ' + str(res[i][4]) + '\n')
		ftxt.close()
    f.close()

