import tensorflow as tf
tfk = tf.keras
tf.enable_eager_execution()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data()