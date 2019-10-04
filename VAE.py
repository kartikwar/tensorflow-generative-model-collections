#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

import prior_factory as prior

import json

class VAE(object):
    model_name = "VAE"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, 
    result_dir, log_dir, input_height=28):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_height
        self.output_height = input_height
        self.output_width = input_height

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist' or dataset_name =='documents':
            # parameters


            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            if dataset_name != 'documents':
                self.data_X, self.data_y = load_mnist(self.dataset_name)
            else:
                self.c_dim = 3
                self.data_X, self.data_y = load_docs(self.dataset_name, self.input_height)



            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    # Gaussian Encoder
    def encoder(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
        with tf.variable_scope("encoder", reuse=reuse):

            #height/2
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            tf.add_to_collection("encoder_conv1", net)
            tf.add_to_collection("encoder_strides1", [2, 2])

            #series of convolutions
            net = slim.repeat(net, 3, slim.conv2d, 128, [4, 4], 
                          trainable=is_training, scope='en_conv2', stride=(2,2))
            # net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = lrelu(bn(net, is_training=is_training, scope='en_b2'))
            tf.add_to_collection("encoder_conv2", net)
            tf.add_to_collection("encoder_strides2", (2, 2))
            tf.add_to_collection("conv2_repititions", 3)
            
            #input shape of net is [bs, curr_height, curr_width, 128], 
            # output shape of net is [bs, curr_height * curr_width *128]
            net = tf.reshape(net, [self.batch_size, -1])
            #need to optimize this for larger images, taking to much space
            # input shape of net is  [bs, curr_height * curr_width *128]
            net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            net_before_gauss = tf.print('shape of net is ', tf.shape(net))
            
            with tf.control_dependencies([net_before_gauss]):
                gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])

        return mean, stddev

    # Bernoulli decoder
    def decoder(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse):
            deconv1_shape = tf.get_collection('encoder_conv2')[0]
            deconv1_shape = deconv1_shape.get_shape().as_list()

            deconv1_strides = tf.get_collection('encoder_strides2')[0]
            deconv1_repitions = tf.get_collection('conv2_repititions')[0]

            deconv2_strides = tf.get_collection('encoder_strides1')[0]
            deconv2_shape = tf.get_collection('encoder_conv1')[0]
            deconv2_shape = deconv2_shape.get_shape().as_list()
            # deconv2_repitions = 


            net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))

            #need to optimize this taking too much space
            net = tf.nn.relu(bn(linear(net, 128 * int(deconv1_shape[1]) * int(deconv1_shape[2]), scope='de_fc2'), is_training=is_training, scope='de_bn2'))
            #height/6
            net = tf.reshape(net, [self.batch_size, int(deconv1_shape[1]), int(deconv1_shape[2]), 128])

            #series of deconvolutions to convert to height/2
            #height/2
            net =  tf.nn.relu( bn(slim.repeat(net, deconv1_repitions, slim.conv2d_transpose, 64, [3, 3], 
                          trainable=is_training, scope='de_dc3', stride=(2,2)), is_training=is_training, scope='de_bn3'))

            # net = tf.nn.relu(
            #     bn(deconv2d(net, [self.batch_size, int(deconv2_shape[1]), int(deconv2_shape[2]), 64], 4, 4, deconv1_strides[0], 
            #     deconv1_strides[1], name='de_dc3'), is_training=is_training,
            #        scope='de_bn3'))
            
            #height
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, self.output_height, self.output_width, 1], 4, 4, 
            deconv2_strides[0], deconv2_strides[1], name='de_dc4'))

            # out = slim.repeat

            return out

    def inference(self): 
        bs = self.batch_size
        self.mu = tf.placeholder(tf.float32, [bs, self.z_dim], name='mu')
        self.sigma = tf.placeholder(tf.float32, [bs, self.z_dim], name='sigma')
        # z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        z = self.mu 
        self.images = self.decoder(z, is_training=False, reuse=True)
        # self.images = self.decoder(z, is_training=False, reuse=True)

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        tf.add_to_collection("input", self.inputs)

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        tf.add_to_collection("z", self.z)

        """ Loss Function """
        # encoding
        self.mu, sigma = self.encoder(self.inputs, is_training=True, reuse=False)

        tf.add_to_collection("mu", self.mu)
        
        tf.add_to_collection("sigma", sigma)        

        self.sigma = sigma

        # sampling by re-parameterization technique
        z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)

        z_shape = tf.print('shape of z is ', tf.shape(z))

        # decoding
        with tf.control_dependencies([z_shape]):
            out = self.decoder(z, is_training=True, reuse=False)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out),
                                            [1, 2])/ 1000.0
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -self.neg_loglikelihood - self.KL_divergence

        self.loss = -ELBO

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1)
            gradients = optimizer.compute_gradients(self.loss, var_list=t_vars, aggregation_method=2)
            self.optim = optimizer.apply_gradients(gradients)
            # self.optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
            #           .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)

        # self.images = self.decoder(z, is_training=False, reuse=True)

        #

        """ Summary """
        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        loss_sum = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        # self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = load_batch(self.dataset_name, self.input_height, idx*self.batch_size, (idx+1)*self.batch_size)
                # batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op, self.loss, self.neg_loglikelihood, self.KL_divergence],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                # self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss, nll_loss, kl_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
        # self.visualize_results()

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def feature_analysis(self, mu, sigma, batch_images):
        # samples = self.sess.run(self.images , feed_dict={self.mu: mu, self.sigma: sigma})
        image_vectors = {}

        mu_max = np.max(mu)

        mu_min = np.min(mu)

        for j in range(np.shape(mu)[1]):

            original_samples = self.sess.run(self.images , feed_dict={self.mu: mu, self.sigma: sigma})

            indexes_to_keep = [ a for a in list(range(j)) ]

            # mu_ = mu[:, indexes_to_keep]

            # mu_feat = mu[0, :]

            mu_ = np.copy(mu)

            

            mu_[:, :] = 0

            mu_[:, indexes_to_keep] = mu[:, indexes_to_keep] 
            # mu_[:, j+1] = mu_min

            samples = self.sess.run(self.images , feed_dict={self.mu: mu_, self.sigma: sigma})

        # samples = self.sess.run(self.images, feed_dict={self.z: z})

            for i in range(len(samples)):
                sample = samples[i]*255
                ori_sample = original_samples[i]*255
                img = batch_images[i]*255
                cv2.imwrite('infer/' + str(j) + '_' + str(i) + '.jpg', sample)
                cv2.imwrite('orinfer/' + str(j) + '_' + str(i) + '.jpg', ori_sample)
                cv2.imwrite('input/' + str(j) + '_'  + str(i) + '.jpg', img)
                x  = np.copy(mu)
                x = x.tolist()
                image_vectors[str(j) + '_' + str(i)] = x[i]
        

        with open('template_vectors.json', 'w') as tv:
            json.dump(image_vectors, tv)

    def visualize_results(self):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = prior.gaussian(self.batch_size, self.z_dim)
    

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})



        # save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #             check_folder(
        #                 self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        # """ learned manifold """
        # if self.z_dim == 2:
        #     assert self.z_dim == 2

        z_tot = None
        id_tot = None

        # for idx in range(0, 100):
            #randomly sampling
        id = np.random.randint(0,self.num_batches)
        id = 100
        batch_images = self.data_X[id * self.batch_size:(id + 1) * self.batch_size]
        batch_labels = self.data_y[id * self.batch_size:(id + 1) * self.batch_size]

        mu, sigma = self.sess.run([self.mu, self.sigma], feed_dict={self.inputs: batch_images})

        # tf.reset_default_graph()

        self.inference()

        self.feature_analysis( mu, sigma, batch_images)


        # samples = self.sess.run(self.images , feed_dict={self.mu: mu, self.sigma: sigma})



        # # samples = self.sess.run(self.images, feed_dict={self.z: z})

        # for i in range(len(samples)):
        #     sample = samples[i]*255
        #     img = batch_images[i]*255
        #     cv2.imwrite('infer/' + str(i) + '.jpg', sample)
        #     cv2.imwrite('input/' + str(i) + '.jpg', img)


        # images = self.sess.run

        # learned_images = self.decoder 

        # if idx == 0:
        #     z_tot = z
        #     id_tot = batch_labels
        # else:
        #     z_tot = np.concatenate((z_tot, z), axis=0)
        #     id_tot = np.concatenate((id_tot, batch_labels), axis=0)

        # save_scattered_image(z_tot, id_tot, -4, 4, name=check_folder(
        #     self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_learned_manifold.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim, self.input_height)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
