import tensorflow as tf
import pprint
tf.enable_eager_execution()
tfk = tf.keras

from resnet_model import resnet18, resnet34, CNN

def main():

    def loss(model, x, y):
        out = model(x)
        y_oh = tf.one_hot(y, depth=100)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_oh, logits=out))

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, model.trainable_variables)

    (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
      (tf.cast(x_train/255.0, tf.float32),
       tf.cast(y_train,tf.int64)))
    dataset = dataset.shuffle(100).batch(32)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
    model = resnet18()

    for epoch in range(2):
        for (batch, (images, labels)) in enumerate(dataset.take(400)):
            grads = grad(model, images, labels)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

    test_res = model(tf.cast(x_test/255.0, tf.float32))
    prec = tf.equal(tf.argmax(test_res, axis=1, output_type=tf.int32), y_test)
    train_acc = tf.reduce_mean(tf.cast(prec, tf.float32))
    print(train_acc)

if __name__ == '__main__':
    main()
