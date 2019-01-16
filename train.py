import tensorflow as tf
import pprint
tf.enable_eager_execution()
tfk = tf.keras

from resnet_model import ResNet, CNN

def main():

    def loss(model, x, y):
        out = model(x)
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, model.variables)

    (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
      (tf.cast(x_train/255, tf.float32),
       tf.cast(y_train,tf.int64)))
    dataset = dataset.shuffle(100).batch(32)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    model = CNN()

    for epoch in range(1):
        print(epoch)
        for (batch, (images, labels)) in enumerate(dataset.take(400)):
            grads = grad(model, images, labels)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())


if __name__ == '__main__':
    main()