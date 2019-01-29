#!/usr/bin/env python
# coding: utf-8

# # Auto Poster Generation
# ## 当前任务
# ### 打分器（实际上可看作二分类问题）
# #### 1. 输入
# - 正样本：已有的海报图像，label为1
# - 负样本：在现有海报图像的基础上随机搭配，label为0
# 
# #### 2. 网络结构
# - 可以用现有的卷积基模型，可能需要fine-tune
# - 也可以自己构造一个简单的模型（尝试）
# - 对于卷积基提取的特征，后面接上Flatten和Dense层，最后做一个二分类
# 
# #### 3. 输出
# - 输出的概率值既可以看作是打分器的分数

# In[1]:


import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


# In[2]:


standard_width = 200
standard_height = 280


# ### 读取数据集并进行预处理

# In[3]:


import glob
import numpy as np
from PIL import Image

poster_positive = glob.glob('./data/poster_positive/*.png')
poster_negative = glob.glob('./data/poster_negative/*.png')
np.random.shuffle(poster_positive)
np.random.shuffle(poster_negative)
poster_positive_num = len(poster_positive)
poster_negative_num = len(poster_negative)

print("poster positive num: " + str(poster_positive_num))
print("poster negative num: " + str(poster_negative_num))


# In[4]:


import random

num_train_positive = 280
num_train_negative = 297

num_validation_positive = 20
num_validation_negative = 36

num_test_positive = 35
num_test_negative = 35

# tuple(number, height, width, dimension)
X_train = np.empty((num_train_positive + num_train_negative, standard_height, standard_width, 3))
Y_train = np.empty((num_train_positive + num_train_negative, 1))

X_validation = np.empty((num_validation_positive + num_validation_negative, standard_height, standard_width, 3))
Y_validation = np.empty((num_validation_positive + num_validation_negative, 1))

X_test = np.empty((num_test_positive + num_test_negative, standard_height, standard_width, 3))
Y_test = np.empty((num_test_positive + num_test_negative, 1))

# make train set
for i in range(num_train_positive):
    im = Image.open(poster_positive[i])
    X_train[i] = np.asarray(im.convert('RGB'), dtype='float64') / 255.0  
    Y_train[i] = 1
    
for i in range(num_train_negative):
    im = Image.open(poster_negative[i])
    X_train[num_train_positive + i] = np.asarray(im.convert('RGB'), dtype='float64') /255.0
    Y_train[num_train_positive + i] = 0
    
# index = [i for i in range(len(X_train))]
# random.shuffle(index)
# X_train = X_train[index]
# Y_train = Y_train[index]

# shuffle the whole train set
zipped = list(zip(X_train, Y_train))
np.random.shuffle(zipped)
X_train[:], Y_train[:] = zip(*zipped)

print(np.array(X_train).shape, np.array(Y_train).shape)
assert len(X_train) == len(Y_train)

# make validation set
for i in range(num_validation_positive):
    im = Image.open(poster_positive[num_train_positive + i])
    X_validation[i] = np.asarray(im.convert('RGB'), dtype='float64') / 255.0  
    Y_validation[i] = 1
    
for i in range(num_validation_negative):
    im = Image.open(poster_negative[num_train_negative + i])
    X_validation[num_validation_positive + i] = np.asarray(im.convert('RGB'), dtype='float64') /255.0
    Y_validation[num_validation_positive + i] = 0

print(np.array(X_validation).shape, np.array(Y_validation).shape)
assert len(X_validation) == len(Y_validation)

# make test set
for i in range(num_test_positive):
    im = Image.open(poster_positive[num_train_positive + num_validation_positive + i])
    X_test[i] = np.asarray(im.convert('RGB'), dtype='float64') / 255.0  
    Y_test[i] = 1
    
for i in range(num_test_negative):
    im = Image.open(poster_negative[num_train_negative + num_validation_negative + i])
    X_test[num_test_positive + i] = np.asarray(im.convert('RGB'), dtype='float64') /255.0
    Y_test[num_test_positive + i] = 0

print(np.array(X_test).shape, np.array(Y_test).shape)
assert len(X_test) == len(Y_test)


# ### 尝试使用VGG16卷积基预训练模型

# In[5]:


# 将VGG16卷积基实例化
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(standard_height, standard_width, 3))


# In[6]:


conv_base.summary()


# In[7]:


# 在卷积基上添加一个密集链接分类器
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1,activation='relu'))


# In[8]:


model.summary()


# In[9]:


# 冻结卷积基
conv_base.trainable = False
print(len(model.trainable_weights))


# In[10]:


# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[11]:


# 训练模型
history = model.fit(X_train,
                    Y_train,
                    epochs=20,
                    batch_size=40,
                    validation_data=(X_validation, Y_validation))

model.save('./model_save/vgg16_plain.h5')


# In[ ]:


results = model.evaluate(X_test, Y_test)
print(results)


# In[ ]:


model.predict(X_test)


# ### VGG16预训练模型效果不好，接下来重新训练一个以Xception为卷积基的网络

# In[ ]:


from keras.applications import Xception
from keras import models
from keras import layers

conv_base_xception = Xception(include_top=False,
                              input_shape=(standard_height, standard_width, 3))


# In[ ]:


conv_base_xception.summary()


# In[ ]:


# 在xception卷积基上添加一个密集链接分类器

model_xception = models.Sequential()
model_xception.add(conv_base_xception)
model_xception.add(layers.Flatten())
model_xception.add(layers.Dense(256, activation='relu'))
model_xception.add(layers.Dense(1,activation='sigmoid'))


# In[ ]:


model_xception.summary()


# In[ ]:


print(len(model_xception.trainable_weights))


# In[ ]:


# 编译模型
model_xception.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['accuracy']) 


# In[ ]:


# 训练模型
from keras.backend import get_session

get_session().run(tf.global_variables_initializer())

history_xception = model_xception.fit(X_train,
                                      Y_train,
                                      epochs=20,
                                      batch_size=16,
                                      validation_data=(X_validation, Y_validation))

model_xception.save('scorer_xception.h5')


# ### 尝试加入style matrix(gram matrix)，这里还是使用VGG16，因为只包含卷积和池化基本操作

# In[ ]:


# 将VGG16卷积基实例化，这次不含参数
from keras.applications import VGG16

conv_base_vgg16 = VGG16(include_top=False,
                        input_shape=(standard_height, standard_width, 3))


# In[ ]:


conv_base_vgg16.summary()


# In[ ]:


# style matrix(gram matrix)

from keras import backend as K

# def gram_matrix(A):
#     """
#     Argument:
#     A -- matrix of shape (n_C, n_H*n_W)
    
#     Returns:
#     GA -- Gram matrix of A, of shape (n_C, n_C)
#     """
    
#     GA = K.dot(A, K.transpose(A))
    
#     return GA

def compute_layer_style(a_S):
#     """
#     Arguments:
#     a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 

#     Returns: 
#     GS -- Gram matrix of S, of shape (n_C, n_C)        
#     """
    
    GS = K.batch_dot(a_S, a_S, axes=[1, 1])
    
    return GS


# In[ ]:


# 构建网络

from keras.models import Model
from keras import layers
from keras import Input

def model_vgg16_style(input_shape):
    
    X_input = Input(input_shape)
    
    X = conv_base_vgg16(X_input)
    
    m, n_H, n_W, n_C = X.get_shape().as_list()
    
    X = layers.core.Reshape([n_H*n_W, n_C])(X)
    
    X = layers.core.Lambda(compute_layer_style)(X)
    
    X = layers.Flatten()(X)
    
    X = layers.Dense(256, activation='relu')(X)
    
    X = layers.Dense(256, activation='relu')(X)
    
    X = layers.Dense(1, activation='sigmoid')(X)
    
    model_vgg16_style = Model(X_input, X)
    
    return model_vgg16_style


# In[ ]:


model_vgg16_style = model_vgg16_style((standard_height, standard_width, 3))


# In[ ]:


model_vgg16_style.summary()


# In[ ]:


# 编译模型
model_vgg16_style.compile(optimizer='rmsprop',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])


# In[ ]:


# 训练模型
from keras.backend import get_session

get_session().run(tf.global_variables_initializer())

history_vgg16_style = model_vgg16_style.fit(X_train,
                                            Y_train,
                                            epochs=5,
                                            batch_size=16,
                                            validation_data=(X_validation, Y_validation))

model_vgg16_style.save('scorer_vgg16_style.h5')


# ### 可能海报提取成高维特征后，就会不收敛，下面尝试搭建一个只带一次卷积池化的简单网络

# In[ ]:


from keras.models import Model
from keras import layers
from keras import Input

def Simple_Model(input_shape):
    
    X_input = Input(input_shape)
        
    X = layers.ZeroPadding2D((3, 3))(X_input)
        
    X = layers.Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = layers.BatchNormalization(axis = 3, name = 'bn0')(X)
    X = layers.Activation('relu')(X)
        
    X = layers.MaxPooling2D((2, 2), name='max_pool')(X)
        
    X = layers.Flatten()(X)
    X = layers.Dense(1, activation='sigmoid', name='fc')(X)
        
    simple_model = Model(inputs = X_input, outputs = X, name='Simple_Model')
    
    
    return simple_model


# In[ ]:


simple_model = Simple_Model((standard_height, standard_width, 3))


# In[ ]:


simple_model.summary()


# In[ ]:


simple_model.compile(optimizer = "Adam",
                     loss = "binary_crossentropy",
                     metrics = ["accuracy"])


# In[ ]:


history_simple = simple_model.fit(X_train,
                                  Y_train,
                                  epochs=5,
                                  batch_size=16,
                                  validation_data=(X_validation, Y_validation))


# ### 尝试只用全连接层

# In[ ]:


from keras.models import Model
from keras import layers
from keras import Input

def dense_model(input_shape):
    
    X_input = Input(input_shape)
    
#     m, n_H, n_W, n_C = X_input.get_shape().as_list()
    
#     X = layers.core.Reshape([n_H*n_W, n_C])(X_input)
    
#     X = layers.core.Lambda(compute_layer_style)(X)
    
    X = layers.Flatten()(X_input)
    
    X = layers.Dense(512, activation='relu')(X)
    
    X = layers.Dense(512, activation='relu')(X)
    
    X = layers.Dense(1, activation='sigmoid')(X)
    
    dense_model = Model(inputs = X_input, outputs = X)
    
    return dense_model


# In[ ]:


dense_model = dense_model((standard_height, standard_width, 3))


# In[ ]:


dense_model.summary()


# In[ ]:


dense_model.compile(optimizer = "Adam",
                    loss = "binary_crossentropy",
                    metrics = ["accuracy"])


# In[ ]:


dense_simple = dense_model.fit(X_train,
                               Y_train,
                               epochs=5,
                               batch_size=16,
                               validation_data=(X_validation, Y_validation))

