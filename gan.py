import cv2
import glob
import keras
import keras.backend as K
import math
import numpy as np
import random
import sys
import tensorflow as tf

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import *
from keras.layers import *
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11
zed = 100


def load_dataset():
    DIR=sys.argv[1]

    train_files = [p for p in glob.glob("./{}/*.png".format(DIR))]

    image_width = 128
    image_height = 128

    channels = 3
    nb_classes = 1

    x_t = np.ndarray(shape=(len(train_files), 
                                image_height, 
                                image_width, 
                                channels),
                     dtype=np.float32)

    i = 0
    for _file in train_files:
        img = load_img(_file)  # this is a PIL image
        img.thumbnail((image_width, image_height))

        x = img_to_array(img)  
        x = x.reshape((128, 128, 3))

        x = (x - 128.0) / 128.0

        x_t[i] = x
        i += 1
        if i % 250 == 0:
            print("%d images to array" % i)
    
    return x_t, None, None, None

xt,yt,xv,yv = load_dataset()

def relu(i):
    return LeakyReLU(.2)(i)

def bn(i):
    return BatchNormalization()(i)

def gen2(): # generative network, 2
    inp = Input(shape=(zed,))
    i = inp
    i = Reshape((1,1,zed))(i)

    ngf=24

    def deconv(i,nop,kw,oh,ow,std=1,tail=True,bm='same'):
        global batch_size
        i = Deconvolution2D(nop,
                            kw,
                            kw,
                            subsample=(std, std),
                            border_mode=bm,
                            output_shape=(batch_size, oh, ow, nop))(i)
        if tail:
            i = bn(i)
            i = relu(i)

        return i

    i = deconv(i, nop=ngf*32, kw=4, oh=4, ow=4, std=1, bm='valid')
    i = deconv(i, nop=ngf*16, kw=4, oh=8, ow=8, std=2)
    i = deconv(i, nop=ngf*8, kw=4, oh=16, ow=16, std=2)
    i = deconv(i, nop=ngf*4, kw=4, oh=32, ow=32, std=2)
    i = deconv(i, nop=ngf*2, kw=4, oh=64, ow=64, std=2)
    i = deconv(i, nop=ngf*1, kw=4, oh=128, ow=128, std=2)

    i = deconv(i,nop=3,kw=4,oh=32,ow=32,std=1,tail=False) # out : 128x128
    
    i = Activation('tanh')(i)

    m = Model(input=inp, output=i)
    return m

def concat_diff(i):
    bv = Lambda(lambda x: K.mean(K.abs(x[:] - 
                                 K.mean(x,axis=0)), axis=-1, keepdims=True))(i)
    i = concatenate([i,bv])

    return i

def dis2():
    inp = Input(shape=(128,128,3))
    i = inp

    ndf=24

    def conv(i, nop, kw, std=1, usebn=True, bm='same'):
        i = Convolution2D(nop, kw, kw, border_mode=bm, subsample=(std, std))(i)
        if usebn:
            i = bn(i)
        i = relu(i)
        return i

    i = conv(i, ndf*1, 4, std=2, usebn=False)
    i = concat_diff(i)
    i = conv(i, ndf*2, 4, std=2)
    i = concat_diff(i)
    i = conv(i, ndf*4, 4, std=2)
    i = concat_diff(i)
    i = conv(i, ndf*8, 4, std=2)
    i = concat_diff(i)
    i = conv(i, ndf*16, 4, std=2)
    i = concat_diff(i)
    i = conv(i, ndf*32, 4, std=2)
    i = concat_diff(i)

    i = Convolution2D(1, 2, 2, border_mode='valid')(i)

    i = Activation('linear', name='conv_exit')(i)
    i = Activation('sigmoid')(i)

    i = Reshape((1,))(i)

    m = Model(input=inp, output=i)
    return m

print('generating G...')
gm = gen2()
gm.summary()

print('generating D...')
dm = dis2()
dm.summary()

def gan(g,d):
    noise = Input(shape=g.input_shape[1:])
    real_data = Input(shape=d.input_shape[1:])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data)

    def log_eps(i):
        return K.log(i+1e-11)

    dloss = - K.mean(log_eps(1-gscore) + .1 * 
                     log_eps(1-rscore) + .9 * log_eps(rscore))
    gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr, b1 = 1e-4, .2 
    optimizer = Adam(lr,beta1=b1)

    grad_loss_wd = optimizer.compute_gradients(dloss, d.trainable_weights)
    update_wd = optimizer.apply_gradients(grad_loss_wd)

    grad_loss_wg = optimizer.compute_gradients(gloss, g.trainable_weights)
    update_wg = optimizer.apply_gradients(grad_loss_wg)

    def get_internal_updates(model):
        inbound_nodes = model._inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [d,g]]

    print('other_parameter_updates for the models:')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    learning_phase = K.learning_phase()

    def gan_feed(sess,batch_image,z_input):
        nonlocal train_step, losses, noise, real_data, learning_phase

        res = sess.run([train_step, losses], feed_dict={
            noise: z_input,
            real_data: batch_image,
            learning_phase: True,
        })

        loss_values = res[1]
        return loss_values

    return gan_feed

print('generating GAN...')
gan_feed = gan(gm,dm)

def r(ep=10000,noise_level=.01):
    sess = K.get_session()

    np.random.shuffle(xt)
    shuffled_cifar = xt
    length = len(shuffled_cifar)

    for i in range(ep):
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)

        j = i % int(length/batch_size)
        minibatch = shuffled_cifar[j*batch_size:(j+1)*batch_size]

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))

        losses = gan_feed(sess,minibatch,z_input)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 10==0: show()

def autoscaler(img):
    limit = 400.
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(img.shape[0])
    for s in scales:
        if s>=imgscale:
            imgscale=s
            break

    img = cv2.resize(img,dsize=(int(img.shape[1]*imgscale),
                     int(img.shape[0]*imgscale)),
                     interpolation=cv2.INTER_NEAREST)

    return img,imgscale

def flatten_multiple_image_into_image(arr):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = int(num+1)
    height = int(math.sqrt(patches)*0.9)
    width = int(patches/height+1)

    img = np.zeros((height*uh+height, width*uw+width, 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index>=num-1:
                break
            channels = arr[index]
            img[row*uh+row:row*uh+uh+row,col*uw+col:col*uw+uw+col,:] = channels
            index+=1

    img,imgscale = autoscaler(img)

    return img,imgscale
import uuid

generation = 0
def show(save=False):
    global generation
    save = "./{}-{}.png".format(format(generation, '03'), str(uuid.uuid4()))
    p = sys.argv[1].split('/')[-1]
    model_path = "./{}-{}.model".format(format(generation, '03'), p)
    i = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
    gened = gm.predict([i])

    gened *= 0.5
    gened +=0.5

    im,ims = flatten_multiple_image_into_image(gened)
    cv2.imshow('gened scale:'+str(ims),im)
    cv2.waitKey(1)

    if save != False:
        cv2.imwrite(save, im*255)
        if generation % 100 == 0:
            gm.save(model_path)
    generation += 1
    
r()