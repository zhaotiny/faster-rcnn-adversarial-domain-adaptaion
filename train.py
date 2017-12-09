# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""
import pdb
from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
import pdb
import re

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()
           # pdb.set_trace()
            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})
         #   pdb.set_trace()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)
       # pdb.set_trace()
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})
    def recover_train(self, sess):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()
           # pdb.set_trace()
            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 / np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 / self.bbox_stds})

       # print "haha"
    def snapshot_npy(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()
           # pdb.set_trace()
            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        data = {}
        all_variables = tf.get_collection(tf.GraphKeys.VARIABLES)
       # pdb.set_trace()
        for i in range(len(all_variables)):
            temp_name = all_variables[i].name
            #print temp_name

            parts = temp_name.split('/');
            if (len(parts) == 1):
                continue
            if (len(parts) == 2):
                scope_name = parts[0]
            if (len(parts) == 3):
                scope_name = parts[0] + '/' + parts[1]
                vari_name_temp = parts[-1].split(':')[0]
                if (vari_name_temp != "weights" and vari_name_temp != "biases"):
                    continue
            if (len(parts) > 3):
                continue
            vari_name = parts[-1].split(':')[0]
            if (scope_name not in data.keys()):
                data[scope_name] = {}
            data[scope_name][vari_name] = sess.run(all_variables[i])
        filename = self.output_dir + '/' + str(iter + 1) + '.npy'
        np.save(filename,data);
       # pdb.set_trace()


        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]

        smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

        conf_score = self.net.get_output('conf_prob')
        conf_label = tf.reshape(self.net.get_output('roi-data')[-1],[-1])
        conf_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=conf_score, labels=conf_label))

        conf_loss = -tf.reduce_mean(tf.log(conf_score))

        loss_source = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + conf_loss
        loss_target = conf_loss
        loss_domain = conf_cross_entropy



        all_variables_trained = tf.get_collection(tf.GraphKeys.VARIABLES)
        #pdb.set_trace()
        ####### extract the list of variables beside those in the domain confusion network
        num_var = len(all_variables_trained)
        normal_variables = []
        print "normal variabls:   "
        for i in range(num_var):
            temp_name = all_variables_trained[i].name
            #print temp_name
            if (re.search(r'fc6_d', temp_name) != None):
                continue
            if (re.search(r'fc7_d', temp_name) != None):
                continue

            if (re.search(r'conf_score', temp_name) != None):
                continue
            normal_variables.append(all_variables_trained[i])
            print temp_name
        ######## extract list of variabls of the domain confusion network
        print "domain variables: "
        domain_variables = []
        for i in range(num_var):
            temp_name = all_variables_trained[i].name
            #print temp_name
            if (re.search(r'fc6_d', temp_name) != None):
                print temp_name
                domain_variables.append(all_variables_trained[i])
                continue
            if (re.search(r'fc7_d', temp_name) != None):
                print temp_name
                domain_variables.append(all_variables_trained[i])
                continue

            if (re.search(r'conf_score', temp_name) != None):
                print temp_name
                domain_variables.append(all_variables_trained[i])
                continue
        print "num of all variables: ", len(all_variables_trained)
        print "num of normal varibales: ", len(normal_variables)
        print "num of domain variables: ", len(domain_variables)
        # final loss
       # loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op_source = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_source, var_list=normal_variables)
        train_op_target = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_target, var_list=normal_variables)
       # train_op_domain = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_domain, global_step=global_step, var_list=all_variables_trained)
        train_op_domain = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_domain, global_step=global_step, var_list=domain_variables)
      #  train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step, var_list=sub_variables)
        # intialize variables
        sess.run(tf.global_variables_initializer())
  	if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            if self.pretrained_model.endswith('.npy'):
                self.net.load(self.pretrained_model, sess, self.saver, True)
                self.recover_train(sess)
            elif self.pretrained_model.endswith('.ckpt'):
                self.saver.restore(sess, self.pretrained_model)
                self.recover_train(sess)
            else:
                print("Unknown model type! Weights not loaded.")
       # if self.pretrained_model is not None:
       #     print ('Loading pretrained model '
       #            'weights from {:s}').format(self.pretrained_model)
      #      self.net.load(self.pretrained_model, sess, True)

        last_snapshot_iter = -1
        timer = Timer()
        #pdb.set_trace()
        for iter in range(max_iters):
            # get one batch
            blobs = data_layer.forward()
         #   number = np.ran
            blobs['is_source'] = np.array([0], dtype='f')

            # Make one SGD update
           # pdb.set_trace()
            feed_dict_source={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes'], self.net.is_source: blobs['is_source']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()
           # pdb.set_trace()
            #pdb.set_trace()                                                                                                                                                                                                                                                                                                        pdb.set_trace()
          #  print sess.run(global_step)
          #  conf_loss_value, _ = sess.run([conf_loss, train_op_target], feed_dict=feed_dict_source,  options=run_options,run_metadata=run_metadata)

            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value,conf_loss_value, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, conf_loss, train_op_source],
                                                                                                feed_dict=feed_dict_source,
                                                                                                options=run_options,
                                                                                                 run_metadata=run_metadata)


          #  print sess.run(global_step)

          #  pdb.set_trace()
            feed_dict_domain={self.net.data: blobs['data'],self.net.keep_prob: 0.5, self.net.im_info: blobs['im_info'], self.net.is_source: blobs['is_source'], self.net.gt_boxes: blobs['gt_boxes']}
            conf_cross_entropy_value, _ = sess.run([conf_cross_entropy, train_op_domain],
                                                        feed_dict=feed_dict_domain,
                                                        options=run_options,
                                                        run_metadata=run_metadata)
         #   pdb.set_trace()
           # pdb.set_trace()
            #print sess.run(global_step)
            timer.toc()
            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, conf_loss: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, conf_loss_value, lr.eval())
                print 'iter: %d / %d, conf_cross_entropy: %.4f'%\
                        (iter+1, max_iters, conf_cross_entropy_value)
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            #    print "bbox pred: ", sess.run(all_variables_trained[38])
            #    print "bbox pred: ", sess.run(all_variables_trained[39])
            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)
                self.snapshot_npy(sess, iter)
             #   pdb.set_trace()
                '''
                data = {}
                all_variables = tf.get_collection(tf.GraphKeys.VARIABLES)
                for i in range(len(all_variables)):
                    temp_name = all_variables[i].name
                  #  print temp_name

                    parts = temp_name.split('/');
                    if (len(parts) == 1):
                        continue
                    if (len(parts) == 2):
                        scope_name = parts[0]
                    if (len(parts) == 3):
                        scope_name = parts[0] + '/' + parts[1]
                        vari_name_temp = parts[-1].split(':')[0]
                        if (vari_name_temp != "weights" and vari_name_temp != "biases"):
                            continue
                    if (len(parts) > 3):
                        continue


                    vari_name = parts[-1].split(':')[0]
                    if (scope_name not in data.keys()):
                        data[scope_name] = {}
                    data[scope_name][vari_name] = sess.run(all_variables[i])
                    print temp_name
                np.save('output/faster_rcnn_end2end/voc_2007_trainval/' + str(iter + 1) + '.npy',data);
                '''



        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)
            self.snapshot_npy(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        #pdb.set_trace()
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V1)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
