import datetime
import glob
import pandas as pd
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
from scipy.misc.pilutil import imread,imresize

from keras.models import *
from keras.layers import *
from keras.layers.merge import *
from keras.optimizers import *
from keras import backend as K
K.set_image_data_format('channels_first')

train_real_data_dir = './Training/Real/*'
train_white_data_dir = './Training/White/*'

real_list = glob.glob(train_real_data_dir)
train_real_data_list = []
train_real_data_list.extend(real_list)

white_list = glob.glob(train_white_data_dir)
train_white_data_list = []
train_white_data_list.extend(white_list)

img_row = img_col = 128
channels = 1
def dis(input_shape):
    def conv_block(input, filters, is_first_layer=False, is_last_layer=False):
        if not is_last_layer:
            x = Conv2D(filters=filters, kernel_size=(4,4), strides=(2,2), padding='same', activation=LeakyReLU(alpha=0.2))(input)
        else:
            x = Conv2D(filters=filters, kernel_size=(4,4), strides=(1,1), padding='same')(input)
        if not is_first_layer:
            x = BatchNormalization()(x)
        return x

    img_A = Input(input_shape)
    img_B = Input(input_shape)
    combined_img = Concatenate(axis=1)([img_A,img_B])
    x = conv_block(combined_img, 8, is_first_layer=True)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 1, is_last_layer=True)

    model = Model([img_A,img_B], x)
    return model

def gen(input_shape):
    def conv_block(input, filters, is_first_layer=False, is_last_layer=False):
        if not is_last_layer:
            x = Conv2D(filters=filters, kernel_size=(4,4), strides=(2,2), padding='same', activation=LeakyReLU(alpha=0.2))(input)
        if not is_first_layer:
            x = BatchNormalization()(x)
        return x

    def deconv_block(input, skip_input, filters):
        x = UpSampling2D(size=(2,2))(input)
        x = Conv2D(filters=filters, kernel_size=(4,4), strides=(1,1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Concatenate(axis=1)([x, skip_input])
        return x

    img_A = Input(input_shape)
    x1 = conv_block(img_A, 4, is_first_layer=True)
    x2 = conv_block(x1, 8)
    x3 = conv_block(x2, 16)
    x4 = conv_block(x3, 32)
    x5 = conv_block(x4, 32)
    d1 = deconv_block(x5, x4, 32)
    d2 = deconv_block(d1, x3, 16)
    d3 = deconv_block(d2, x2, 8)
    d4 = deconv_block(d3, x1, 4)
    d5 = UpSampling2D(size=(2,2))(d4)
    out_img = Conv2D(1, name='out_img', kernel_size=(4,4), strides=(1,1), activation='tanh', padding='same')(d5)
    model =  Model(img_A, out_img)
    return model


input_shape=(channels,img_row,img_col)
crop_shape=(img_row,img_col)
G = gen(input_shape)
D = dis(input_shape)


D_optimizer = Adam(lr=0.0002, beta_1=0.5)
D.trainable=False
D.compile(loss='mse', optimizer=D_optimizer,metrics=['accuracy'])
D.summary()

AM_optimizer = Adam(lr=0.0002, beta_1=0.5)
img_A = Input(input_shape)
img_B = Input(input_shape)
fake_A = G(img_B)
valid = D([fake_A,img_B])
AM = Model([img_A,img_B],[valid,fake_A])
AM.compile(loss=['mse','mae'],loss_weights=[1,1],optimizer=AM_optimizer)
AM.summary()

def generator_training_Img(real_list_dir,white_list_dir,resize=None,batch_size=32):
    batch_real_img=[]
    batch_white_img=[]
    for _ in range(batch_size):
        random_index = np.random.randint(len(real_list_dir))
        real_img = imread(real_list_dir[random_index], mode='L')
        white_img = imread(white_list_dir[random_index], mode='L')
        if resize:
            real_img = imresize(real_img,resize)
            white_img = imresize(white_img,resize)
        batch_real_img.append(real_img)
        batch_white_img.append(white_img)
    batch_real_img = np.array(batch_real_img)/127.5-1
    batch_real_img = np.expand_dims(batch_real_img,axis=1)
    batch_white_img = np.array(batch_white_img)/127.5-1
    batch_white_img = np.expand_dims(batch_white_img,axis=1)
    return batch_real_img,batch_white_img



batch_size=128
all_epoch=3000
valid = np.ones((batch_size,channels,8,8))
fake  = np.zeros((batch_size,channels,8,8))

start_time=datetime.datetime.now()


for now_iter in range(all_epoch):
    ori_img,white_img = generator_training_Img(real_list_dir=train_real_data_list,
                                               white_list_dir=train_white_data_list,
                                               resize=(img_row,img_col),
                                               batch_size=batch_size)
    fake_A = G.predict(white_img)
    
    
    D_loss_Real = D.train_on_batch([ori_img,white_img],valid)
    D_loss_Fake = D.train_on_batch([ori_img,fake_A],fake)
    D_loss = 0.5 * np.add(D_loss_Real,D_loss_Fake)
    
    G_loss = AM.train_on_batch([ori_img, white_img], [valid, ori_img])

    end_time = datetime.datetime.now() - start_time
    print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss1: %f,loss2: %f] [time:%s]" % (now_iter,all_epoch,D_loss[0],D_loss[1]*100,G_loss[0],G_loss[1],end_time))


import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
import glob
import numpy as np
import pandas as pd
from scipy.misc.pilutil import imread,imresize,imsave
from keras.models import load_model

def generator_test_Img(list_dir,resize):
    output_training_img=[]
    for i in list_dir:
        img = imread(i,mode='L')
        img = imresize(img,resize)
        output_training_img.append(img)
    output_training_img = np.array(output_training_img)/127.5-1
    output_training_img = np.expand_dims(output_training_img,axis=1)
    return output_training_img

def numpy_to_csv(input_image,image_number=10,save_csv_name='predict.csv'):
    save_image=np.zeros([int(input_image.size/image_number),image_number],dtype=np.float32)

    for image_index in range(image_number):
        save_image[:,image_index]=input_image[image_index,:,:].flatten()

    base_word='id'
    df = pd.DataFrame(save_image)
    index_col=[]
    for i in range(n):
        col_word=base_word+str(i)
        index_col.append(col_word)
    df.index.name='index'
    df.columns=index_col
    df.to_csv(save_csv_name)
    print("Okay! numpy_to_csv")


test_data_dir=r'./Test/White/*'
test_data_dir_list=glob.glob(test_data_dir)
test_data_list=[]
test_data_list.extend(test_data_dir_list)

n=10
output_img_col = output_img_row=128
white_img = generator_test_Img(list_dir=test_data_list,resize=(output_img_col,output_img_row))
image_array = G.predict(white_img).squeeze(1)
image_array = (image_array+1)/2


print(image_array.shape)

numpy_to_csv(input_image=image_array,image_number=n,save_csv_name='Predict_%d_%d.csv' % (batch_size,all_epoch))
















