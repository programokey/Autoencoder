import tensorflow as tf
from tensorflow.contrib import layers
from DataFetch import MNISTDataSet
import numpy as np
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1
SAMPLE_SPACE_DIM = 10
REAL_COUNT = 100
FAKE_COUNT = 100
N_critic = 20
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

with tf.name_scope('inputs'):
    input_data = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name='input_data')
    random_sample = tf.placeholder(tf.float32, [None, SAMPLE_SPACE_DIM],
                                   name='sample')
    discriminator_labels = tf.placeholder(tf.float32, [None], 'discriminator_labels')

HIDDEN_DIM = 1000
with tf.variable_scope('generator'):
    fc = layers.fully_connected(random_sample, HIDDEN_DIM)
    for i in range(5):
        fc = tf.layers.batch_normalization(layers.fully_connected(fc, HIDDEN_DIM) + fc)
    fc = layers.fully_connected(fc, IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS, activation_fn=tf.nn.sigmoid)
    generated_output = tf.reshape(fc, (-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    # generator_outsize = 128
    # conv = tf.reshape(layers.fully_connected(random_sample, IMG_WIDTH*IMG_HEIGHT*1), [-1, IMG_WIDTH, IMG_HEIGHT, 1])
    # prev_output = tf.layers.conv2d(conv, filters=generator_outsize, kernel_size=(3, 3), padding='same',
    #                                    activation=tf.nn.relu)
    # for i in range(5):
    #     conv = tf.layers.conv2d(conv, filters=generator_outsize, kernel_size=(3, 3), padding='same',
    #                             activation=tf.nn.relu)
    #     conv = tf.layers.batch_normalization(conv + prev_output)
    #     prev_output = conv
    # generated_output = tf.layers.conv2d(conv, filters=IMG_CHANNELS, kernel_size=(1, 1), activation=tf.nn.sigmoid)

def get_discriminator(discriminator_input, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        discriminator_outsize = 128
        conv = tf.layers.batch_normalization(discriminator_input)
        prev_output = tf.layers.conv2d(conv, filters=discriminator_outsize, kernel_size=(3, 3), padding='same',
                                       activation=tf.nn.relu)
        for i in range(5):
            conv = tf.layers.conv2d(conv, filters=discriminator_outsize, kernel_size=(3, 3), padding='same',
                                    activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv + prev_output)
            prev_output = conv
        conv = tf.layers.conv2d(conv, filters=2, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        fc = tf.reshape(conv, [-1, IMG_WIDTH * IMG_HEIGHT * 2])
        prediction = layers.fully_connected(fc, 1, activation_fn=tf.sigmoid)
        return prediction,fc

real_prediction, _= get_discriminator(input_data)
fake_prediction,discriminator_fc = get_discriminator(generated_output, reuse=True)

with tf.variable_scope('discriminator', reuse=True):
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator"):
        v_name = v.name[len('discriminator/'):-2]
        v = tf.get_variable(v_name, constraint=lambda x: tf.clip_by_value(x, -0.5, 0.5))

with tf.variable_scope('loss'):
    discriminator_loss = tf.reduce_mean(fake_prediction) - tf.reduce_mean(real_prediction)
    generative_loss = tf.reduce_mean(fake_prediction)

    discriminator_summ = tf.summary.merge(
        [tf.summary.scalar('Wasserstein_Distance', -discriminator_loss)])
    generator_summ = tf.summary.merge([tf.summary.scalar('generative_loss', generative_loss),
                                       tf.summary.scalar('generated_output_square_mean',
                                                         tf.reduce_mean(tf.square(generated_output)))])
    img_summ = tf.summary.merge([tf.summary.image('generated_imgs', generated_output)])

with tf.variable_scope('train'):
    global_step = tf.Variable(0)
    update_step = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=100, decay_rate=0.95, staircase=True)

    discriminator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    discriminator_train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(discriminator_loss, var_list=discriminator_variables)

    generator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    generative_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    generated_output_gradients, discriminator_fc_gradients = generative_optimizer.compute_gradients(loss=generative_loss, var_list=[generated_output, discriminator_fc])
    gradient_summ = tf.summary.merge([tf.summary.scalar('square_generated_output_gradients', tf.reduce_mean(tf.square(generated_output_gradients))),
                                      tf.summary.scalar('abs_generated_output_gradients',tf.reduce_mean(tf.abs(generated_output_gradients))),
                                      tf.summary.scalar('square_discriminator_fc_gradients', tf.reduce_mean(tf.square(discriminator_fc_gradients))),
                                      tf.summary.scalar('abs_discriminator_fc_gradients',tf.reduce_mean(tf.abs(discriminator_fc_gradients))),])

    generative_train_step = generative_optimizer.minimize(generative_loss,var_list=generator_variables)
    tf.summary.scalar('learning rate', learning_rate)
init = tf.global_variables_initializer()

if __name__ == '__main__':
    dataset = MNISTDataSet(batch_size=REAL_COUNT, epochs_num=100)
    test_imgs = dataset.all_test_data()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('log', sess.graph)
        sess.run(init)
        step = 0
        generate_step = 0
        labels = np.concatenate((np.ones(REAL_COUNT, dtype=np.float32), np.zeros(FAKE_COUNT, dtype=np.float32)), axis=0)
        for imgs, _, epoch in dataset.training_batches():
            print('step>>>', step)
            # z = np.random.normal(0, 1, size=(FAKE_COUNT, SAMPLE_SPACE_DIM))
            z = np.random.uniform(-1, 1, size=(FAKE_COUNT, SAMPLE_SPACE_DIM))
            _, summ, r_pred, f_pred = sess.run((discriminator_train_step, discriminator_summ, real_prediction, fake_prediction),
                                     feed_dict={input_data: imgs,
                                                random_sample: z})
            print(r_pred.reshape(-1))
            print(f_pred.reshape(-1))
            writer.add_summary(summ, step)
            if step % N_critic == 0 and step > 0:
                _, summ, grad_summ = sess.run((generative_train_step, generator_summ, gradient_summ),
                                              feed_dict={random_sample: np.random.uniform(-1, 1,
                                                                                          size=(FAKE_COUNT,
                                                                                                SAMPLE_SPACE_DIM))})
                writer.add_summary(summ, generate_step)
                writer.add_summary(grad_summ, generate_step)
                generate_step += 1

                if generate_step % 20 == 0:
                    generated_imgs, summ = sess.run((generated_output, img_summ), feed_dict={random_sample: z})
                    writer.add_summary(summ, step)
                    show_images(np.reshape(generated_imgs[:100, :], (100, -1)), pic_name='img/generate_%d' % step)

            step += 1
            sess.run(update_step)
