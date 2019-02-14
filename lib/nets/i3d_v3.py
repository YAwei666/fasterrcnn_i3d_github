# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.network import Network


class Unit3D(snt.AbstractModule):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        """Connects the module to inputs.

        Args:
          inputs: Inputs to the Unit3D component.
          is_training: whether to use training mode for snt.BatchNorm (boolean).

        Returns:
          Outputs from the module.
        """
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,
                         use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net


class i3d(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'vgg_16'

    def _image_to_head(self, is_training, reuse=None):
        self._final_endpoint=0
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = self._image
            # 时间维度上复制
            net = tf.expand_dims(net, 0)
            net = tf.tile(net, multiples=[1, 20, 1, 1, 1])
            end_points = {}
            end_point = 'Conv3d_1a_7x7'
            net = Unit3D(output_channels=64, kernel_shape=[7, 7, 7],
                         stride=[2, 2, 2], name=end_point)(net, is_training=is_training)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points
            end_point = 'MaxPool3d_2a_3x3'
            net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                                   padding=snt.SAME, name=end_point)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points
            end_point = 'Conv3d_2b_1x1'
            net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                         name=end_point)(net, is_training=is_training)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points
            end_point = 'Conv3d_2c_3x3'
            net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                         name=end_point)(net, is_training=is_training)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points
            end_point = 'MaxPool3d_3a_3x3'
            net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                                   padding=snt.SAME, name=end_point)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_3b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_3c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'MaxPool3d_4a_3x3'
            net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                                   padding=snt.SAME, name=end_point)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_4f'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
                self._layers['I3D_feature'] = net
                print(self._layers)
                print(net.shape)
                net=tf.slice(net,[0,0,0,0,0],[-1,1,-1,-1,-1])
            end_points[end_point] = net
            print(net.shape)
            net=tf.squeeze(net,axis=[1])
            print(net.shape)
        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def _head_to_tail_backup(self, pool5, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            pool5_flat = slim.flatten(pool5, scope='flatten')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                                   scope='dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                                   scope='dropout7')

        return fc7

    def _head_to_tail(self, pool5, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            end_point = 'MaxPool3d_5a_2x2'
            end_points={}
            net=pool5
            net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                                   padding=snt.SAME, name=end_point)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0a_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            end_points[end_point] = net
            if self._final_endpoint == end_point: return net, end_points

            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                with tf.variable_scope('Branch_1'):
                    branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_1,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_2'):
                    branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0a_1x1')(net, is_training=is_training)
                    branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                      name='Conv3d_0b_3x3')(branch_2,
                                                            is_training=is_training)
                with tf.variable_scope('Branch_3'):
                    branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                                strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                                name='MaxPool3d_0a_3x3')
                    branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                      name='Conv3d_0b_1x1')(branch_3,
                                                            is_training=is_training)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

            pool5_flat = slim.flatten(net, scope='flatten')
            self.print_tensor_infomation(pool5_flat,'pool5_flat')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                                   scope='dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                                   scope='dropout7')

        return fc7


    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = {}

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            v_name='/'.join(v.name.split('/')[1:])
            v_name=v_name.replace(':0','')
            v_name='RGB/inception_i3d/'+v_name
            if v_name in var_keep_dic:
                if 'ixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance' in v_name:
                    print(v.shape)
                print(v.name)
                print(v_name)
                variables_to_restore[v_name]=v

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                              self._scope + "/fc7/weights": fc7_conv,
                                              self._scope + "/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
