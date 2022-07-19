
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import Input, Model,Sequential
from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense

# 打开数据
fp = "./RML2016.10a_dict.pkl"
Xd = pickle.load(open(fp, 'rb'), encoding='latin')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

#  Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
# 70%样本用于训练，30%样本用于测试
np.random.seed(2020)
n_examples = X.shape[0]
n_train = n_examples * 0.7
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = to_onehot(trainy)
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))


#进一步提取信号的幅度值和相位值
from numpy import linalg as la
def amp_phase(a):
    b = np.zeros(a.shape)
    n = int(a.shape[1]/2)
    for i in range(n):
        x_tran = a[:,i*2,:] + 1j*a[:,i*2+1,:]
        b[:,i*2,:] = np.abs(x_tran)
        b[:,i*2+1,:] = np.arctan2(a[:,i*2,:], a[:,i*2+1,:]) / np.pi
    return b

#同样设置为另一个像素点上的两个通道的值
def norm(a,b):
    for i in range(a.shape[0]):
        norm_amp = 1 / la.norm(a[i, 0, :], 2)
        b[i,:,:] = b[i,:,:] * norm_amp
        for j in range(int(a.shape[1]/2)):
            a[i,j*2,:] = a[i,j*2,:] * norm_amp
    return a,b

X_train1a = amp_phase(X_train)
X_test1a = amp_phase(X_test)
X_train1a,X_train1 = norm(X_train1a,X_train)
X_test1a,X_test1 = norm(X_test1a,X_test)

Xtrain = np.concatenate((X_train,X_train1a),axis=1)
Xtest = np.concatenate((X_test,X_test1a),axis=1)

in_shp = list(Xtrain.shape[1:])

from tensorflow.keras.layers import Convolution1D, Dense, Activation, Dropout, Flatten, MaxPooling2D
import tensorflow.keras.backend as K

from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, \
    Concatenate, Conv2D, Add, Activation, Lambda

''' 通道注意力机制：
    对输入feature map进行spatial维度压缩时，作者不单单考虑了average pooling，
    额外引入max pooling作为补充，通过两个pooling函数以后总共可以得到两个一维矢量。
    global average pooling对feature map上的每一个像素点都有反馈，而global max pooling
    在进行梯度反向传播计算只有feature map中响应最大的地方有梯度的反馈，能作为GAP的一个补充。
'''

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


''' 空间注意力机制:
    还是使用average pooling和max pooling对输入feature map进行压缩操作，
    只不过这里的压缩变成了通道层面上的压缩，对输入特征分别在通道维度上做了
    mean和max操作。最后得到了两个二维的feature，将其按通道维度拼接在一起
    得到一个通道数为2的feature map，之后使用一个包含单个卷积核的隐藏层对
    其进行卷积操作，要保证最后得到的feature在spatial维度上与输入的feature map一致，
'''

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    # 实验验证先通道后空间的方式比先空间后通道或者通道空间并行的方式效果更佳
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, )

    return cbam_feature

from tensorflow.keras.layers import  LSTM, MaxPooling1D
ConvInput = Input(in_shp)

x1 = Convolution1D(256, 3, padding='same', data_format='channels_first',
                    activation='relu')(ConvInput)

x1 = Dropout(rate=0.3)(x1)
x2 = Convolution1D(256, 3, padding='same', data_format='channels_first',
                   activation='relu')(x1)
x2 = Dropout(rate=0.3)(x2)

x2 = Convolution1D(80, 3, padding='same', data_format='channels_first',
                    activation='relu')(x2)

x2 = Dropout(rate=0.3)(x2)
x2 = Convolution1D(80, 3, padding='same', data_format='channels_first',
                   activation='relu')(x2)
x2 = Dropout(rate=0.3)(x2)

x = LSTM(128, return_sequences=True, recurrent_dropout=0.3)(x2)
x = LSTM(128,  dropout=0.3)(x)

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

x = K.reshape(x, (-1, 1, 1, 128))
x = cbam_block(x)      #添加混合注意力机制
x = Flatten()(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(11)(x)
x_out = Activation('softmax')(x)
model = Model(inputs=ConvInput, outputs=x_out, name='CBAMLSTM')
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])


batch_size = 256
epochs = 100
history = model.fit(Xtrain, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(Xtest, Y_test)
                    )

# Show loss curves
print(history.history['loss'])
print('')
print(history.history['val_loss'])
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.show()
acc = {}

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    iters = np.reshape([[[i, j] for j in range(len(labels))] for i in range(len(labels))], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, round(cm[i, j], 3), va='center', ha='center', fontsize=7)  # 显示对应的数字

    plt.tight_layout()
    plt.ylabel('True label', fontsize=8)
    plt.xlabel('Predicted label', fontsize=8)

# Plot confusion matrix
acc_list = []
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_SNRs = list(test_SNRs)
    test_X_i = Xtest[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)

    conf = np.zeros([len(mods), len(mods)])
    confnorm = np.zeros([len(mods), len(mods)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(mods)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    plot_confusion_matrix(confnorm, labels=mods, title="CBAMLSTM Confusion Matrix (SNR=%d)" % (snr))
    filepath = 'C:/Users/admin/Desktop/TCN-with-attention-master/data/picture1/CBAMLSTM/' + "ConvNet Confusion Matrix (SNR=%d)" % (snr) + 'SNR.png'
    plt.savefig(filepath)
    plt.show()

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor

    acc[snr] = 1.0 * cor / (cor + ncor)
    print('snr:', snr)
    print('acc:', acc[snr])
    # zhi = round(acc[snr], 4)
    acc_list.append(acc[snr])

import pandas as pd

df1 = pd.DataFrame({'acc':acc_list})
df1.to_csv('C:/Users/admin/Desktop/TCN-with-attention-master/data/datatcn1/CBAMLSTM.csv', index=False)

plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CBAM+LSTM Classification Accuracy on RadioML 2016.10 Alpha")
plt.yticks(np.linspace(0, 1, 6))
plt.show()