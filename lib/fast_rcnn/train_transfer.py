# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""
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

    def __init__(self, sess, saver, network, source_imdb, target_imdb,
                 source_roidb, target_roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.source_imdb = source_imdb
        self.target_imdb = target_imdb
        self.source_roidb = source_roidb
        self.target_roidb = target_roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(source_roidb)
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

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

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

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 / np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 / self.bbox_stds})

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
            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        data = {}
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for i in range(len(all_variables)):
            variable_name = all_variables[i].name

            parts = variable_name.split('/');
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
        source_data_layer = get_data_layer(self.source_roidb,
                                                  self.source_imdb.num_classes)
        target_data_layer = get_data_layer(self.target_roidb,
                                                  self.target_imdb.num_classes)

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

        # Domain classification
        conf_score = self.net.get_output('conf_score')
        conf_prob = self.net.get_output('conf_prob')

        # Mean prob of all positive and negative examples (ROIs)
        mean_conf_score = tf.reshape(tf.reduce_mean(conf_score, 0), [-1, 2])
        mean_conf_prob = tf.reshape(tf.reduce_mean(conf_prob, 0), [-1, 2])
        pos_mean_prob = mean_conf_prob[0][0]
        neg_mean_prob = mean_conf_prob[0][1]

        #This bit here is to produce a domain label vector of the right size.
        num_rois = tf.shape(conf_score)[0]
        #num_rois = tf.Print(num_rois, [num_rois], "Number of rois: ")
        conf_label = tf.cond(
                tf.equal(self.net.is_source[0], 1), # check if source
                lambda: tf.ones([num_rois], dtype=tf.int32), # ones if source
                lambda: tf.zeros([num_rois], dtype=tf.int32)) # zeros if target

        ### use the mean of all the ROI to compute the loss
        conf_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = conf_score, labels=conf_label))

        # Loss
        conf_loss = - tf.reduce_mean(tf.log(conf_prob))

        loss_source = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + conf_loss
        loss_target = conf_loss
        loss_domain = conf_cross_entropy

        # Summaries
        tf.summary.scalar("loss_source", loss_source)
        tf.summary.scalar("loss_target", loss_target)
        tf.summary.scalar("loss_cross_entropy", conf_cross_entropy)
        tf.summary.scalar("loss_pos", pos_mean_prob)
        tf.summary.scalar("loss_neg", neg_mean_prob)

        summary_merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("tensorboard_test/", sess.graph)

        ## Extract the list of variables
        all_variables_trained = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        num_var = len(all_variables_trained)
        normal_variables = []
        domain_variables = []
        for i in range(num_var):
            variable_name = all_variables_trained[i].name
            if (re.search(r'fc6_d', variable_name) != None):
                domain_variables.append(all_variables_trained[i])
                continue
            if (re.search(r'fc7_d', variable_name) != None):
                domain_variables.append(all_variables_trained[i])
                continue
            if (re.search(r'conf_score', variable_name) != None):
                domain_variables.append(all_variables_trained[i])
                continue
            normal_variables.append(all_variables_trained[i])

        print "num of all variables: ", len(all_variables_trained)
        print "num of normal varibales: ", len(normal_variables)
        print "num of domain variables: ", len(domain_variables)

        # Optimizers and learning rate
        global_step = tf.Variable(0, trainable=False)


        ## different learning rate for generative and adversive network
        lr_g = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)

        lr_d = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE , global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op_source = tf.train.MomentumOptimizer(lr_g, momentum).minimize(loss_source, global_step=global_step, var_list=normal_variables)
        train_op_target = tf.train.MomentumOptimizer(lr_g, momentum).minimize(loss_target, var_list=normal_variables)
        train_op_domain = tf.train.MomentumOptimizer(lr_d, momentum).minimize(loss_domain, var_list=domain_variables)
        #pdb.set_trace()


        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model weights from {:s}').format(self.pretrained_model)
            if self.pretrained_model.endswith('.npy'):
                self.net.load(self.pretrained_model, sess, self.saver, ignore_missing=True)
                self.recover_train(sess)
            elif self.pretrained_model.endswith('.ckpt'):
                self.saver.restore(sess, self.pretrained_model)
                self.recover_train(sess)
            else:
                print("Unknown model type! Weights not loaded.")

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            if iter % 2 == 0:
                blobs = source_data_layer.forward()
                blobs['is_source'] = np.array([1], dtype='f')
            else:
                blobs = target_data_layer.forward()
                blobs['is_source'] = np.array([0], dtype='f')


            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes'], self.net.is_source: blobs['is_source']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()
            ### test the domain module only
            if iter % 2 == 0:
                '''
                rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, conf_loss_value, _ = \
                        sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, conf_loss, train_op_source],
                                feed_dict=feed_dict,
                                options=run_options,
                                run_metadata=run_metadata)
                '''
                rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, conf_loss_value, pos_mean_score_source = \
                        sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, conf_loss, pos_mean_prob],
                                feed_dict=feed_dict,
                                options=run_options,
                                run_metadata=run_metadata)
                pdb.set_trace()
                conf_cross_entropy_value_source, pos_mean_score_domain1, _ = \
                    sess.run([conf_cross_entropy, pos_mean_prob, train_op_domain],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                pdb.set_trace()
            else:
                '''
                loss_target_value, _ = \
                        sess.run([conf_loss, train_op_target],
                                feed_dict=feed_dict,
                                options=run_options,
                                run_metadata=run_metadata)
                '''
                loss_target_value, pos_mean_score_target = \
                        sess.run([conf_loss, pos_mean_prob],
                                feed_dict=feed_dict,
                                options=run_options,
                                run_metadata=run_metadata)

                conf_cross_entropy_value_target, pos_mean_score_domain2, _ = \
                    sess.run([conf_cross_entropy, pos_mean_prob, train_op_domain],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
              #  pdb.set_trace()



            timer.toc()
            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                ### Add print the output of the predicted probability of the domain classifier
                print 'iter: %6d / %6d, source: total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, conf_loss_source: %.4f, pos_mean_score_source: %.4f, lr: %f' % \
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value + conf_loss_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, conf_loss_value, pos_mean_score_source, lr_g.eval())
                print '                       target: conf_loss_target: %.4f, pos_mean_score_target: %.4f, lr: %f'%( loss_target_value, pos_mean_score_target,lr_g.eval())
                print '                       domain: conf_cross_entropy_source: %.4f, conf_cross_entropy_target: %.4f, pos_mean_score_dsource: %.4f, pos_mean_score_dtarget: %.4f,lr: %f'%(conf_cross_entropy_value_source, conf_cross_entropy_value_target, pos_mean_score_domain1, pos_mean_score_domain2, lr_d.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            #    print "bbox pred: ", sess.run(all_variables_trained[38])
            #    print "bbox pred: ", sess.run(all_variables_trained[39])

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
             #   self.snapshot(sess, iter)
                self.snapshot_npy(sess, iter)

            if (iter) % cfg.TRAIN.SUMMARY_ITERS == 0:
                summary_results = sess.run(summary_merged, feed_dict=feed_dict)
                summary_writer.add_summary(summary_results, iter + 1)


        if last_snapshot_iter != iter:
          #  self.snapshot(sess, iter)
            self.snapshot_npy(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
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


def train_net(network, source_imdb, target_imdb, source_roidb, target_roidb,
              output_dir, pretrained_model=None, max_iters=40000):
    """Train a Faster R-CNN network with domain transfer."""
    source_roidb = filter_roidb(source_roidb)
    target_roidb = filter_roidb(target_roidb)
    saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
   # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sw = SolverWrapper(sess, saver, network, source_imdb, target_imdb,
                           source_roidb, target_roidb, output_dir,
                           pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
