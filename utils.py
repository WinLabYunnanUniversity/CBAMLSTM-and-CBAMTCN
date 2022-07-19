# necessary python libraries
import numpy as np
import pickle
import math
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Dense,LSTM,concatenate,Convolution1D,Dropout,Flatten,Reshape
from tensorflow.keras.models import Model


#%%
def get_pf(x_val,y_val,val_SNRs,model,epoch,pf_min,pf_max):
    '''
        callback for pfs evaluation at evert epoch end
    '''

    y_val_hat = model.predict(x_val,verbose=0)
    cm = confusion_matrix(np.argmax(y_val, 1), np.argmax(y_val_hat, 1))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    pf = 100 * cm_norm[1][0]
    print(cm_norm)
    print(pf)
    print("False Alarm:%.3f%%" % pf)
    # # set the pf stop interval for a CFAR detector
    if (pf > pf_min) & (pf < pf_max):
        print("Pf meet the threshold, training stopped")
        model.stop_training = True

def performance_evaluationH1H0(save_path,x_test,y_test,test_SNRs,model):
    '''
        Evaluate final model's performance
    '''
    y_test_hat = model.predict(x_test, verbose=1)
    plt, pf = getConfusionMatrixPlot(np.argmax(y_test, 1), np.argmax(y_test_hat, 1))
    pd_list = []
    snrs = np.linspace(-18, 18, 19)
    snrs = np.array(snrs, dtype='int16')
    for snr in snrs:
        test_x_i = x_test[np.where(test_SNRs == snr)]
        test_y_i = y_test[np.where(test_SNRs == snr)]
        test_y_i_hat = np.array(model.predict(test_x_i, verbose=0))
        cm = confusion_matrix(np.argmax(test_y_i, 1), np.argmax(test_y_i_hat, 1))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        pd_list.append(cm_norm[0][0])
        # save Pds result to xls file, the last element if Pf
    import csv
    pd_list.append(pf)
    with open(save_path, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(pd_list)

def getFontColor(value):
    '''
        set color in confusion matrix plot
    '''
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def getConfusionMatrixPlot(true_labels, predicted_labels):
    '''
        plot confusion matrix
    '''
    import matplotlib.pyplot as plt
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)
    # add color bar
    plt.colorbar(res)
    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))
    # add genres as ticks
    alphabet = ['with signal','noise only'] 
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    plt.title('Confusion matrix for all snrs')
    print(" Miss Detection:%.3f%%"%(100*cm_norm[0][1]))
    print(" False Alarm:%.3f%%"%(100*cm_norm[1][0]))
    return plt,100*cm_norm[1][0]

def radioml_IQ_dataH1H0(filename, mod_name, swap_dim = False):
    '''
        load dataset for single node model training
    '''
    snrs = ""
    mods = ""
    lbl = ""
    Xd = pickle.load(open(filename, 'rb'), encoding='latin')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)
    lbl = np.array(lbl)

    #信噪比为-18到18dB的数据
    X1 = []
    lbl1 = []
    snrs1 = np.linspace(-18, 18, 19)
    snrs1 = np.array(snrs1, dtype='int16')
    for mod in mods:
        for snr in snrs1:
            X1.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl1.append((mod, snr))
    X1 = np.vstack(X1)
    lbl1 = np.array(lbl1)

    SNR = []
    noise_signal = []
    XX = []
    mod_name1 = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    for i in mod_name1:
        index = np.where(lbl == i)[0]
        X01 = X[index]
        lbl01 = lbl[index]

        indexX1 = np.where(lbl1 == i)[0]
        X11 = X1[indexX1]
        lbl11 = lbl1[indexX1]

        for ii in lbl01:   #把所有调制信号命名为signal信号
            ii[0] = "signal"
        lbl01 = lbl01

        for ii in lbl11:
            ii[0] = "signal"
        lbl11 = lbl11

        # mod_name = ['signal']
        index01 = np.where(lbl01 == mod_name)[0]
        X02 = X01[index01]
        lbl02 = lbl01[index01]

        index11 = np.where(lbl11 == mod_name)[0]
        X12 = X11[index11]
        lbl12 = lbl11[index11]

        SNR1 = []
        for item in lbl12:
            SNR1.append(item[-1])
        SNR1 = np.array(SNR1, dtype='int16')

        XX1 = []
        for i in range(X12.shape[0]):
            XX2 = X12[i]
            XX1.append(XX2)
        XX1 = np.array(XX1)

        SNR_name2 = ['-20']   #提取信噪比SNR为-20dB所有信号
        index02 = np.where(lbl02 == SNR_name2)[0]
        X3 = X02[index02]
        lbl3 = lbl02[index02]

        X5 = []
        for j in range(19):
            for jj in range(X3.shape[0]):
                X4 = X3[jj]
                X5.append(X4)
        X5 = np.array(X5)
        noise_signal1 = X5      #信噪比SNR为-20dB所有信号作为噪声负样本

        for j in range(noise_signal1.shape[0]):
            y2 = noise_signal1[j]
            noise_signal.append(y2)

        for j in range(XX1.shape[0]):
            y2 = XX1[j]
            XX.append(y2)

        for j in range(SNR1.shape[0]):
            y2 = SNR1[j]
            SNR.append(y2)

    SNR = np.array(SNR)
    XX = np.array(XX)
    noise_signal = np.array(noise_signal)

    dataset = np.concatenate((XX, noise_signal), axis=0)
    labelset = np.concatenate(([[1, 0]] * len(XX), [[0, 1]] * len(noise_signal)), axis=0)
    labelset = np.array(labelset, dtype='int16')
    # use snr -100 to represent noise samples
    SNR = np.concatenate((SNR, [-100] * len(noise_signal)), axis=0)

    total_num = len(dataset)
    shuffle_idx = np.random.choice(range(0, total_num), size=total_num, replace=False)
    dataset = dataset[shuffle_idx]
    labelset = labelset[shuffle_idx]
    SNR = SNR[shuffle_idx]

    # split the whole dataset with ratio 3:1:1 into training, validation and testing set
    train_num = int(total_num * 0.6)
    val_num = int(total_num * 0.2)

    x_train = dataset[0:train_num]
    y_train = labelset[0:train_num]
    x_val = dataset[train_num:train_num + val_num]
    y_val = labelset[train_num:train_num + val_num]
    x_test = dataset[train_num + val_num:]
    y_test = labelset[train_num + val_num:]
    val_SNRs = SNR[train_num:train_num + val_num]
    test_SNRs = SNR[train_num + val_num:]

    if (swap_dim):
        x_train = np.einsum('ijk->ikj', x_train)
        x_val = np.einsum('ijk->ikj', x_val)
        x_test = np.einsum('ijk->ikj', x_test)

    print("Training data:",x_train.shape)
    print("Training labels:",y_train.shape)
    print("Validation data:",x_val.shape)
    print("Validation labels:",y_val.shape)
    print("Testing data",x_test.shape)
    print("Testing labels",y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, val_SNRs, test_SNRs

#%% other networks
def DNN(lr,input_shape,drop_ratio):
    DenseInput = Input(input_shape)
    x1 = Flatten()(DenseInput)
    x1 = Dense(256,activation='relu')(x1)
    x1 = Dropout(rate=drop_ratio)(x1)
    x2 = Dense(128,activation='relu')(x1)
    x2 = Dropout(rate=drop_ratio)(x2)
    x3 = Dense(64,activation='relu')(x2)
    x3 = Dropout(rate=drop_ratio)(x3)
    predictions = Dense(2, activation='softmax')(x3)
    model = Model(inputs=DenseInput,outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary() 
    return model

def CNN1(lr, input_shape):
    ConvInput = Input(input_shape)
    x1 = Convolution1D(256, 3, padding='same', data_format='channels_first', input_shape=input_shape,
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
    x = Flatten()(x2)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2)(x)
    x_out = Activation('softmax')(x)
    model = Model(inputs=ConvInput, outputs=x_out, name='CBAMLSTM')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model

def lstm(lr,input_shape):
    units = 128  
    num_classes = 2
    lstm_inputs = Input(input_shape)
    x1 = LSTM(units=units,return_sequences=True,recurrent_dropout=0.2)(lstm_inputs)
    x2 = LSTM(units=units, dropout=0.2)(x1)
    predictions = Dense(num_classes, activation='softmax')(x2)
    model = Model(inputs=lstm_inputs,outputs=predictions)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model

def TCNtest(lr, input_shape):
    x_input = Input(input_shape)

    from tcn import TCN
    from tensorflow.keras.layers import Activation
    x = TCN(nb_filters=64,  # 在卷积层中使用的过滤器数。可以是列表。
            kernel_size=3,  # 在每个卷积层中使用的内核大小。
            nb_stacks=1,  # 要使用的残差块的堆栈数。
            dilations=[2 ** i for i in range(6)],  # 扩张列表。示例为：[1、2、4、8、16、32、64]。
            # 用于卷积层中的填充,值为'causal' 或'same'。
            # “causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。
            # “same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
            padding='causal',
            use_skip_connections=True,  # 是否要添加从输入到每个残差块的跳过连接。
            dropout_rate=0.2,  # 在0到1之间浮动。要下降的输入单位的分数。
            return_sequences=False,  # 是返回输出序列中的最后一个输出还是完整序列。
            activation='relu',  # 残差块中使用的激活函数 o = Activation(x + F(x)).
            kernel_initializer='he_normal',  # 内核权重矩阵（Conv1D）的初始化程序。
            use_batch_norm=False,  # 是否在残差层中使用批处理规范化。
            use_layer_norm=False,  # 是否在残差层中使用层归一化。
            # name='tcn' #使用多个TCN时，要使用唯一的名称
            )(x_input)
    # x = Dense(64)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(2)(x)
    x_out = Activation('softmax')(x)
    model = Model(inputs=x_input, outputs=x_out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model


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

def CBAMTCN(lr, input_shape):
    ConvInput = Input(input_shape)
    x1 = Convolution1D(256, 3, padding='same', data_format='channels_first',input_shape=input_shape,
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
    x2 = Flatten()(x2)
    x2 = Reshape(target_shape=(80, input_shape[-1]))(x2)
    x2 = concatenate([x2, ConvInput], axis=1)

    from tcn import TCN
    from tensorflow.keras.layers import Activation
    x = TCN(nb_filters=64,  # 在卷积层中使用的过滤器数。可以是列表。
            kernel_size=3,  # 在每个卷积层中使用的内核大小。
            nb_stacks=1,  # 要使用的残差块的堆栈数。
            dilations=[2 ** i for i in range(6)],  # 扩张列表。示例为：[1、2、4、8、16、32、64]。
            # 用于卷积层中的填充,值为'causal' 或'same'。
            # “causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。
            # “same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
            padding='causal',
            use_skip_connections=True,  # 是否要添加从输入到每个残差块的跳过连接。
            dropout_rate=0.2,  # 在0到1之间浮动。要下降的输入单位的分数。
            return_sequences=False,  # 是返回输出序列中的最后一个输出还是完整序列。
            activation='relu',  # 残差块中使用的激活函数 o = Activation(x + F(x)).
            kernel_initializer='he_normal',  # 内核权重矩阵（Conv1D）的初始化程序。
            use_batch_norm=False,  # 是否在残差层中使用批处理规范化。
            use_layer_norm=False,  # 是否在残差层中使用层归一化。
            # name='tcn' #使用多个TCN时，要使用唯一的名称
            )(x2)
    x = K.reshape(x, (-1, 1, 1, 64))
    x = cbam_block(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2)(x)
    x_out = Activation('softmax')(x)
    model = Model(inputs=ConvInput, outputs=x_out, name='CBAMTCN')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model

def CBAMLSTM(lr, input_shape):
    ConvInput = Input(input_shape)
    x1 = Convolution1D(256, 3, padding='same', data_format='channels_first', input_shape=input_shape,
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
    x2 = Flatten()(x2)
    x2 = Reshape(target_shape=(80, input_shape[-1]))(x2)
    x2 = concatenate([x2, ConvInput], axis=1)

    x = LSTM(128, return_sequences=True, recurrent_dropout=0.2)(x2)
    x = LSTM(128, dropout=0.2)(x)

    from tensorflow.keras.layers import Activation

    x = K.reshape(x, (-1, 1, 1, 128))
    x = cbam_block(x)
    x = Flatten()(x2)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2)(x)
    x_out = Activation('softmax')(x)
    model = Model(inputs=ConvInput, outputs=x_out, name='CBAMLSTM')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model
