import numpy as np

class MNISTDataSet(object):
    def __init__(self, training_set_size=40000, batch_size=128, epochs_num=20):
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
            for i, pixel in enumerate(data):
                imgs[i] = pixel / 255
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

if __name__ == '__main__':
    data = MNISTDataSet()
    print(data.batch_size, data.epochs_num)
    print(data.imgs.shape, data.labels.shape)
    for imgs, labels, epoch in data.training_batches():
        print(imgs.shape, labels.shape, epoch)
        break