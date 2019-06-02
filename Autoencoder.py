import tensorflow as tf
from tensorflow.contrib import layers
from DataFetch import MNISTDataSet
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 128

ENCODE_DIM = 2


def show_images(images, col_size=28, row_size=28, n=10, pic_name='orgin.png'):
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
    # plt.savefig(pic_name)
    plt.show()


with tf.name_scope('inputs'):
    input_data = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_data') #[batch_size, 784]

with tf.name_scope('encoder'):
    out_size = 64
    conv1 = tf.layers.conv2d(inputs=input_data, filters=out_size, kernel_size=(7, 7), strides=(2, 2),
                             activation=tf.nn.relu)
    conv1 = tf.layers.batch_normalization(conv1)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=out_size, kernel_size=(3, 3), strides=(1, 1),
                             activation=tf.nn.relu)
    conv2 = tf.layers.batch_normalization(conv2)

    conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=(3, 3), strides=(1, 1),
                             activation=tf.nn.relu)
    conv3 = tf.layers.batch_normalization(conv3)
    fc = tf.reshape(conv3, (-1, 7*7))
    encoder_output = layers.fully_connected(fc, ENCODE_DIM)

with tf.name_scope('decoder'):
    fc_decode = layers.fully_connected(encoder_output, 7*7)
    fc = tf.reshape(conv3, (-1, 7, 7, 1))
    conv_t1 = tf.layers.conv2d_transpose(fc, filters=out_size, kernel_size=(3, 3), activation=tf.nn.relu)
    conv_t1 = tf.layers.batch_normalization(conv_t1)

    conv_t2 = tf.layers.conv2d_transpose(conv_t1, filters=out_size, kernel_size=(3, 3), activation=tf.nn.relu)
    conv_t2 = tf.layers.batch_normalization(conv_t2)

    conv_t3 = tf.layers.conv2d_transpose(conv_t2, filters=out_size, kernel_size=(8, 8), strides=(2,2), activation=tf.nn.relu)

    reconstruct = tf.layers.conv2d(conv_t3, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                         activation=tf.nn.relu)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(reconstruct - input_data))
    train_summ = tf.summary.merge([tf.summary.scalar('loss', loss)])
    test_summ = tf.summary.merge([tf.summary.scalar('test_loss', loss)])

with tf.name_scope('train'):
    global_step = tf.Variable(0)
    update_step = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=100, decay_rate=0.90, staircase=True)
    # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss + 0.02*regularizer)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar('learning rate', learning_rate)

init = tf.global_variables_initializer()



if __name__ == '__main__':
    dataset = MNISTDataSet()
    test_imgs = dataset.all_test_data()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('log', sess.graph)
        sess.run(init)
        step = 0
        for imgs, labels, epoch in dataset.training_batches():
            feed_dict = {input_data: imgs}
            _, summ = sess.run((train_step,train_summ), feed_dict=feed_dict)
            writer.add_summary(summ, step)

            if step % 30 == 0:
                test_feed_dict = {input_data: test_imgs}
                summ =sess.run(test_summ, feed_dict=test_feed_dict)
                writer.add_summary(summ, step)
            if step % 50 == 0:
                test_feed_dict = {input_data: np.array(test_imgs[0:100])}
                show_images(np.reshape(np.array(test_imgs[0:100]), (100, -1)))
                reconstructed_imgs = sess.run(reconstruct, feed_dict=test_feed_dict)
                show_images(np.reshape(reconstructed_imgs, (100, -1)))
            step += 1
            sess.run(update_step)
