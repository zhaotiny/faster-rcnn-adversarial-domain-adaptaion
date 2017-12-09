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
import re

namepath = '../data/kitti_train.txt'
filepath = '../data/training/image_2/';
imgpath_temp = './results/'
txtpath_temp = './results/'
modelpath = './output/faster_rcnn_end2end/voc_2007_trainval'
CLASSES = ('__background__','car')
CLASSES2 = ('__background__', 'Car')
       #    'aeroplane', 'bicycle', 'bird', 'boat',
        #   'bottle', 'bus', 'car', 'cat', 'chair',
         #  'cow', 'diningtable', 'dog', 'horse',
          # 'motorbike', 'person', 'pottedplant',
          # 'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)
    #pdb.set_trace()
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)


def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo2(sess, net, image_name):
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

    # Visualize detections for each class
 #   im = im[:, :, (2, 1, 0)]
 #   fig, ax = plt.subplots(figsize=(12, 12))
 #   ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    res = np.array([])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
       # cls_group = np.ones(boxes.shape[0]) * cls_ind;
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
       # dets = np.hstack((cls_boxes, cls_group[:, np.newaxis],
        #                  cls_scores[:, np.newaxis])).astype(np.float32)
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
    files = [f for f in os.listdir(modelpath)]
    args = parse_args()
    net = get_network(args.demo_net)
	
    # init session
    ######## test on differnet models with different iterations
    for file in files:
	model = os.path.join(modelpath, file)
	# pdb.set_trace()
    	if (re.search(r'.ckpt$', file) == None): continue
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	config=tf.ConfigProto(allow_soft_placement=True)
	#config.gpu_options.per_process_gpu_memory_fraction = 0.3
	sess = tf.Session(config=config)
	# net = get_network(args.demo_net)
	# load model
	saver = tf.train.Saver()
	args.model = model
	saver.restore(sess, args.model)
	#sess.run(tf.initialize_all_variables())

	print '\n\nLoaded network {:s}'.format(args.model)
	f = open(namepath, 'r')
	testFiles = []
	tempname = re.findall(r'[0-9]+', file)[0]
	#pdb.set_trace()
	imgpath = imgpath_temp + tempname + '/'
	txtpath = txtpath_temp + tempname + '/'
	if (not os.path.exists(imgpath)):
	    os.mkdir(imgpath)
	if (not os.path.exists(txtpath)):
	    os.mkdir(txtpath)
	for line in f:
		name = line.strip('\r\n')
		testFiles.append(name)
	print testFiles[0:10]
	numFile = len(testFiles)
	#numFile = 2
	########################### test on each images
	for i in range(numFile):
	#im_name = filepath + testFiles[i]
		im_name = testFiles[i] + '.png'
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for data/demo/{}'.format(im_name)
		#   demo(sess, net, im_name)
		res = demo2(sess, net, im_name)
		numBox = res.shape[0]
		tofile_txt = txtpath + testFiles[i]+ '.txt'
	#	tofile_img = imgpath + testFiles[i] + '.jpg'
		ftxt = open(tofile_txt,'w')
		#im2 = cv2.imread(filepath + im_name)

		for i in range(numBox):
		    ftxt.write(CLASSES2[int(res[i][5])] + ' ' + str(res[i][0]) + ' ' + str(res[i][1]) + ' ' + str(res[i][2]) + ' ' + str(res[i][3]) + ' ' \
			       + str(res[i][4]) + '\n')
		ftxt.close()
	#pdb.set_trace()
	sess.close()


