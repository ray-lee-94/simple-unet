import keras
import keras.layers as KL
import os
import keras.initializers as KI
import numpy as np
from keras.optimizers import SGD
from keras import callbacks
from generator import DataGenerator,padding,cropping
import glob
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)


def Copy_crop(target_tensor, feature_map):
    target_shape = target_tensor.get_shape().as_list()[1]*2
    feature_shape = feature_map.get_shape().as_list()[1]
    corn1 =int((feature_shape-target_shape)/2)
    return KL.Cropping2D(((corn1,corn1),(corn1,corn1)))


def initial_weight(filters):
    return KI.random_normal(stddev=np.sqrt(2/(filters*9)))


def contract_block(input_tensor, filters=64, stage=1, use_bias=True, train_bn=True):
    conv_name = "contract_stage_" + str(stage) + "conv_"
    bn_name = "contract_stage_" + str(stage) + "bn_"
    if stage > 1:
        input_tensor = KL.MaxPool2D(name="max_pooling_stage_" + str(stage - 1))(input_tensor)
    x = KL.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=initial_weight(filters),
                  use_bias=use_bias, name=conv_name + "1",padding='same')(input_tensor)
    x = BatchNorm(name=bn_name + "1")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=initial_weight(filters),
                  use_bias=use_bias, name=conv_name + "2",padding='same')(x)
    x = BatchNorm(name=bn_name + "2")(x, training=train_bn)
    x = KL.Activation("relu", name="contract_stage_" + str(stage) + "_out")(x)
    return x


def expansive(input_tensor, feature_map, filters=512, stage=1, use_bias=True, train_bn=True):
    conv_name = "expan_stage_" + str(stage) + "conv_"
    bn_name = "expand_stage_" + str(stage) + "bn_"
    x = KL.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2),kernel_initializer=initial_weight(filters),
                           use_bias=use_bias, name=conv_name + "1")(
        input_tensor)  # up conv2x2

    croped_feature = Copy_crop(input_tensor, feature_map)(feature_map)
    x = KL.concatenate([croped_feature, x])
    x = KL.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=initial_weight(filters),padding='same',
                  use_bias=use_bias, name=conv_name + "2")(x)
    x = BatchNorm(name=bn_name + "1")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=initial_weight(filters),padding='same',
                  use_bias=use_bias, name=conv_name + "3")(x)
    x = BatchNorm(name=bn_name + "3")(x, training=train_bn)
    x = KL.Activation("relu", name="expansive_stage" + str(stage) + "_out")(x)
    return x


class UNet():
    def __init__(self,custom_callback=None):
        self.keras_model = self.build()
        self.custom_callback=None

    def build(self):
        inputs = KL.Input(shape=(576, 576, 1), name="UNet")
        stage1 = contract_block(inputs, 64, 1)
        stage2 = contract_block(stage1, 128, 2)
        stage3 = contract_block(stage2, 256, 3)
        stage4 = contract_block(stage3, 512, 4)
        stage5 = contract_block(stage4, 1024, 5)
        # according to the paper ,add drop-out, there is no rate in the paper
        contract_out=KL.Dropout(0.5,name="dropout")(stage5)

        ex_stage4 = expansive(contract_out, stage4, 512, stage=4)
        ex_stage3 = expansive(ex_stage4, stage3, 256, stage=3)
        ex_stage2 = expansive(ex_stage3, stage2, 128, stage=2)
        ex_stage1 = expansive(ex_stage2, stage1, 64, stage=1)

        ex_output=KL.Conv2D(1,kernel_size=(1,1),use_bias=True,name="output")(ex_stage1)
        ex_softmax=KL.Activation('sigmoid',name='act_out')(ex_output)
        model = keras.Model(inputs, ex_softmax, name="unet")

        return model

    def train(self):
        if self.custom_callback is None:
            if not os.path.exists('./snaps'):
                os.mkdir('./snaps')
            self.custom_callback=[callbacks.ModelCheckpoint(filepath="./snaps/weight-best.h5",monitor='acc',
                                                            save_best_only=True),
                                  callbacks.ReduceLROnPlateau(monitor='acc',factor=0.1,patience=4),

                                  ]
        self.keras_model.compile(SGD(lr=1e-2,momentum=0.99,nesterov=True),
                                 loss=keras.losses.binary_crossentropy,
                                 metrics=['accuracy'])
        self.keras_model.fit_generator(DataGenerator(batch_size=4),steps_per_epoch=50,epochs=50,
                                       callbacks=self.custom_callback,)
    def test(self):

        self.keras_model.load_weights('./snaps/weight-best.h5')
        test_path='./images/test/*.*'
        test_images=glob.glob(test_path)
        for image in test_images:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            padded_img = padding(img, target_shape=576)
            padded_img=np.array(padded_img, dtype=np.float32)
            padded_img=np.expand_dims(padded_img,axis=-1)
            predict=self.keras_model.predict_on_batch(np.expand_dims(padded_img,0))[0]
            croped_predict=cropping(predict,target_shape=512)

            cv2.imshow("",croped_predict)
            cv2.waitKey(10000)


unet=UNet()
unet.keras_model.summary()
unet.train()
unet.test()