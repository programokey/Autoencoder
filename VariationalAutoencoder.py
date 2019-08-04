import tensorflow as tf
from tensorflow.contrib import layers
from DataFetch import MNISTDataSet
import numpy as np
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
BATCH_SIZE = 400

ENCODE_DIM = 2

GAUSSIAN_OUTPUT = False

def show_images(images, col_size=28, row_size=28, n=10, pic_name='origin.png'):
    res = np.zeros((n*row_size, n*col_size))
    for i in range(n):
        for j in range(n):
            for row in range(row_size):
                for col in range(col_size):
                    res[i*row_size + row][j*col_size + col] = \
                        images[i*n + j, row*row_size + col]
    pic = (res*255).astype(np.uint8)
    plt.figure()
    plt.imshow(pic)
    plt.savefig(pic_name)
    plt.close()
    # plt.show()

with tf.name_scope('inputs'):
    input_data = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_data') #[batch_size, 784]
    epsilon = tf.placeholder(tf.float32, [None, ENCODE_DIM], name='epsilon') #[batch_size, ENCODE_DIM]
HIDDEN_DIM = 1000
with tf.name_scope('encoder'):
    reshapeed_input_data = tf.reshape(tensor=input_data, shape=(-1, 28 * 28))
    fc = tf.layers.batch_normalization(layers.fully_connected(reshapeed_input_data, HIDDEN_DIM))
    fc = tf.layers.batch_normalization(layers.fully_connected(fc, HIDDEN_DIM))
    fc = tf.layers.batch_normalization(layers.fully_connected(fc, HIDDEN_DIM))
    # out_size = 256
    # conv1 = tf.layers.conv2d(inputs=input_data, filters=out_size, kernel_size=(7, 7), strides=(2, 2),
    #                          activation=tf.nn.relu)
    # conv1 = tf.layers.batch_normalization(conv1)`
    #
    # conv2 = tf.layers.conv2d(inputs=conv1, filters=out_size, kernel_size=(3, 3), strides=(1, 1),
    #                          activation=tf.nn.relu)
    # conv2 = tf.layers.batch_normalization(conv2)
    #
    # conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=(3, 3), strides=(1, 1),
    #                          activation=tf.nn.relu)
    # conv3 = tf.layers.batch_normalization(conv3)
    # fc = tf.reshape(conv3, (-1, 7*7))
    mu = layers.fully_connected(fc, ENCODE_DIM, activation_fn=None)
    log_sigma_square = layers.fully_connected(fc, ENCODE_DIM, activation_fn=tf.nn.sigmoid)


    encoder_sample = epsilon*tf.exp(log_sigma_square/2) + mu
    encoder_output = encoder_sample
    # encoder_output = mu

with tf.name_scope('decoder'):
    fc_decode = tf.layers.batch_normalization(layers.fully_connected(encoder_output, HIDDEN_DIM))
    fc_decode = tf.layers.batch_normalization(layers.fully_connected(fc_decode, HIDDEN_DIM))
    fc_decode = tf.layers.batch_normalization(layers.fully_connected(fc_decode, HIDDEN_DIM))
    # fc_decode = layers.fully_connected(encoder_output, 7*7*3)
    # fc = tf.reshape(fc_decode, (-1, 7, 7, 3))
    # conv_t1 = tf.layers.conv2d_transpose(fc, filters=out_size, kernel_size=(3, 3), activation=tf.nn.relu)
    # conv_t1 = tf.layers.batch_normalization(conv_t1)
    #
    # conv_t2 = tf.layers.conv2d_transpose(conv_t1, filters=out_size, kernel_size=(3, 3), activation=tf.nn.relu)
    # conv_t2 = tf.layers.batch_normalization(conv_t2)
    #
    # conv_t3 = tf.layers.conv2d_transpose(conv_t2, filters=out_size, kernel_size=(8, 8), strides=(2,2), activation=tf.nn.relu)

    if GAUSSIAN_OUTPUT:
        pass
        # reconstruct_mu = tf.layers.conv2d(conv_t3, filters=1, kernel_size=(1, 1), strides=(1, 1),
        #                                  activation=None)
        # reconstruct_log_sigma_square = tf.layers.conv2d(conv_t3, filters=1, kernel_size=(1, 1), strides=(1, 1),
        #                               activation=tf.nn.sigmoid)
        # reconstruct = reconstruct_mu
    else:
        # reconstruct_p = tf.layers.conv2d(conv_t3, filters=1, kernel_size=(1, 1), strides=(1, 1),
        #                                  activation=tf.nn.tanh)
        reconstruct_p = layers.fully_connected(fc_decode, 28*28, activation_fn=tf.nn.sigmoid)
        reconstruct_p = tf.reshape(reconstruct_p, (-1, 28, 28, 1))
        reconstruct = tf.to_float(reconstruct_p > 0.5)
with tf.name_scope('loss'):

    latent_distribution_loss = 0.5 * (
                tf.exp(-log_sigma_square) + tf.square(mu) / tf.exp(log_sigma_square) - ENCODE_DIM + log_sigma_square)
    # latent_distribution_loss = 0.5 * (
    #             tf.exp(-log_sigma_square) + tf.square(mu) / tf.exp(log_sigma_square) + log_sigma_square)
    # latent_distribution_loss = 0.5 * (tf.exp(log_sigma_square) + tf.square(mu) - log_sigma_square -1)
    if GAUSSIAN_OUTPUT:
        # output_dim = reconstruct_mu.shape[-1] * reconstruct_mu.shape[-2] * reconstruct_mu.shape[-3]
        # log_likelihood = -reconstruct_log_sigma_square - tf.square((input_data - reconstruct_mu))/tf.exp(reconstruct_log_sigma_square) - output_dim/2*tf.math.log(2*tf.math.pi)
        log_likelihood = -reconstruct_log_sigma_square - tf.square((input_data - reconstruct_mu)) / tf.exp(
            reconstruct_log_sigma_square)
        # log_likelihood = tf.squared_difference(input_data, reconstruct_mu)

        # loss = tf.reduce_mean(
        #     tf.reduce_sum(-log_likelihood, axis=[-1, -2, -3]))
    else:
        log_likelihood = input_data*tf.log(reconstruct_p + 1e-3) + (1 - input_data)*tf.log(1 - reconstruct_p + 1e-3)
    loss = tf.reduce_mean(tf.reduce_sum(-log_likelihood, axis=[-1, -2, -3]) + tf.reduce_sum(latent_distribution_loss, axis=-1))
    # loss = tf.reduce_mean(tf.squared_difference(reconstruct_p, input_data))
    # loss = tf.reduce_mean(-log_likelihood) + tf.reduce_mean(latent_distribution_loss)
    train_summ = tf.summary.merge([tf.summary.scalar('loss', loss), tf.summary.scalar('log_likelihood', tf.reduce_mean(log_likelihood)), tf.summary.scalar('latent_distribution_loss', tf.reduce_mean(latent_distribution_loss))])
    test_summ = tf.summary.merge([tf.summary.scalar('test_loss', loss)])

with tf.name_scope('train'):
    global_step = tf.Variable(0)
    update_step = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=100, decay_rate=0.85, staircase=True)
    # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss + 0.02*regularizer)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar('learning rate', learning_rate)

init = tf.global_variables_initializer()

if __name__ == '__main__':
    dataset = MNISTDataSet(batch_size=BATCH_SIZE, epochs_num=100)
    test_imgs = dataset.all_test_data()
    from itertools import product
    test_z = np.array([np.linspace(start=-1, stop=1, num=100) for _ in range(ENCODE_DIM)], dtype=np.float32).reshape(-1, ENCODE_DIM)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('log', sess.graph)
        sess.run(init)
        step = 0
        for imgs, labels, epoch in dataset.training_batches():
            # assert not (imgs == 0).all()
            print('step>>>', step)
            reconstruct_pp, _, summ = sess.run((reconstruct_p, train_step, train_summ), feed_dict={input_data: imgs, epsilon: np.random.normal(0, 1, [BATCH_SIZE, ENCODE_DIM])})
            # print('-'*100)
            # print(reconstruct_pp[0].reshape(-1))
            # print('-' * 100)
            # print(imgs[0].reshape(-1))
            # print(mu_out[0])

            # for item in mu_out:
            #     print(item)
            # if (imgs == 0).all():
            #     show_images(np.reshape(np.array(imgs[0:100]), (100, -1)), pic_name='img/origin_%d' % step)
            # assert not (imgs == 0).all()
            writer.add_summary(summ, step)

            if step % 30 == 0:
                summ =sess.run(test_summ, feed_dict={input_data: test_imgs, epsilon: np.random.normal(0, 1, [len(test_imgs), ENCODE_DIM])})
                writer.add_summary(summ, step)
            if step % 200 == 0:
                show_images(np.reshape(np.array(test_imgs[0:100]), (100, -1)), pic_name='img/origin_%d'%step)
                reconstructed_imgs = sess.run(reconstruct_p,
                                                      feed_dict={input_data: np.array(test_imgs[0:100]),
                                                                 epsilon: np.random.normal(0, 1, [100, ENCODE_DIM])})
                print(reconstructed_imgs[0])
                show_images(np.reshape(reconstructed_imgs, (100, -1)), pic_name='img/reconstruct_%d'%step)
            if step % 1000 == 0:
                #encoder_output : [size, 2]
                reconstructed_imgs = sess.run(reconstruct, feed_dict={encoder_output: test_z})
                show_images(np.reshape(reconstructed_imgs, (100, -1)), pic_name='img/generate_%d'%step)

            step += 1
            sess.run(update_step)
