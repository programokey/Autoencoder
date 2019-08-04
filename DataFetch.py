import numpy as np
class MNISTDataSet(object):
    def __init__(self, training_set_size=40000, batch_size=100, epochs_num=20):
        self.training_set_size = training_set_size
        self.epochs_num = epochs_num
        self.batch_size = batch_size
        self.loadImages()

    def loadImages(self, imgs_filename="train-images.idx3-ubyte", labels_filename="train-labels.idx1-ubyte"):
        with open(imgs_filename, 'rb') as f:
            data = f.read()
            magic = int.from_bytes(data[:4], byteorder='big')
            num = int.from_bytes(data[4:8], byteorder='big')
            row_size = int.from_bytes(data[8:12], byteorder='big')
            col_size = int.from_bytes(data[12:16], byteorder='big')
            data = data[16:]
            imgs = np.zeros(num * col_size * row_size)
            assert len(imgs) == len(data)
            for i, pixel in enumerate(data):
                imgs[i] = 1.0 if (pixel / 255) > 0.5 else 0
                # imgs[i] = data[i] / 255
            del data
            self.imgs = imgs.reshape((num, 28, 28, -1))

        with open(labels_filename, 'rb') as f:
            data = f.read()
            magic = int.from_bytes(data[:4], byteorder='big')
            num = int.from_bytes(data[4:8], byteorder='big')
            data = data[8:]
            labels = np.zeros(num, dtype=np.int32)
            for i, label in enumerate(data):
                labels[i] = label
            del data
            self.labels = labels

    def training_batches(self):
        for epoch in range(self.epochs_num):
            for j in range(self.training_set_size//self.batch_size):
                yield self.imgs[j * self.batch_size:(j + 1) * self.batch_size], \
                      self.labels[j * self.batch_size:(j + 1) * self.batch_size], \
                      epoch

    def all_test_data(self):
        return self.imgs[self.training_set_size:self.training_set_size + 200]

    def test_data(self):
        for epoch in range(self.epochs_num):
            for j in range((self.imgs.shape[0] - self.training_set_size)//self.batch_size):
                yield self.imgs[self.training_set_size + j * self.batch_size:self.training_set_size + (j + 1) * self.batch_size], \
                      self.labels[self.training_set_size + j * self.batch_size:self.training_set_size + (j + 1) * self.batch_size], \

import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
    # plt.show()
    plt.close()


if __name__ == '__main__':
    data = MNISTDataSet(batch_size=128,epochs_num=1)
    # # print(data.batch_size, data.epochs_num)
    # # print(data.imgs.shape, data.labels.shape)
    step = 0
    for imgs, labels, epoch in data.training_batches():
        # print(imgs.shape)
        show_images(np.reshape(np.array(imgs[0:100]), (100, -1)), pic_name='img/MNIST_%d.png'%step)
        step += 1
        # print(imgs.shape, labels.shape, epoch)
        # break