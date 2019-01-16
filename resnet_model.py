import tensorflow as tf
tf.enable_eager_execution()
tfk = tf.keras

class BasicBlock(tfk.layers.Layer):

    def __init__(self, filter_num, stride=1, downsample=False, *args, **kwargs):
        super(BasicBlock, self).__init__(*args, **kwargs)
        self.conv1 = tfk.layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tfk.layers.Activation('relu')
        self.conv2 = tfk.layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, padding='same')
        self.bn2 = tfk.layers.BatchNormalization()
        if downsample:
            self.downsample = tfk.Sequential()
            self.downsample.add(tfk.layers.Conv2D(filter_num, kernel_size=(1,1), strides=stride))
            self.downsample.add(tfk.layers.BatchNormalization())
                
        else:
            self.downsample = lambda x: x
        self.stride = stride
    
    def calll(self, input):
        residual = self.downsample(input) 

        conv1 = self.conv1(input)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add = tfk.layers.add([bn2, residual])
        out = self.relu(add)
        return out

    

class ResNet(tfk.Model):

    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        #input size reduced
        self.inplanes = 64
        self.conv1 = tfk.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1))
        self.bn1 = tfk.layers.BatchNormalization()
        self.relu = tfk.layers.Activation('relu')
        self.maxpool = tfk.layers.MaxPool2D(pool_size=(2,2), strides=(1,1))
        self.layer1 = self._build_resblock(block, 64, layers[0])
        self.layer2 = self._build_resblock(block, 128, layers[1], stride=2)
        self.layer3 = self._build_resblock(block, 256, layers[2], stride=2)
        self.layer4 = self._build_resblock(block, 512, layers[3], stride=2)
        self.avgpool = tfk.layers.GlobalAveragePooling2D()
        self.fc = tfk.layers.Dense(100) 
    
    def _build_resblock(self, block, filter_num, blocks, stride=1):
        downsample = stride != 1

        layers = []
        res_blocks = tfk.Sequential()
        res_blocks.add(block(filter_num, stride, downsample))
        for _ in range(1, blocks):
            res_blocks.add(block(filter_num, stride))
        
        return res_blocks
    
    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

class CNN(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)
        self.conv1 = tfk.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1))
        self.bn = tfk.layers.BatchNormalization()
        self.relu = tfk.layers.Activation('relu')
        self.pool = tfk.layers.MaxPool2D(pool_size=(2,2), strides=(1,1))
        self.avgpool = tfk.layers.GlobalAveragePooling2D()
        self.fc = tfk.layers.Dense(100)
        self.softmax = tfk.layers.Softmax()
    
    def call(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.avgpool(x)
        out = self.fc(x)
        return out