


from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import Input, Model,Sequential
from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense

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

from numpy import linalg as la
def amp_phase(a):
    b = np.zeros(a.shape)
    n = int(a.shape[1]/2)
    for i in range(n):
        x_tran = a[:,i*2,:] + 1j*a[:,i*2+1,:]
        b[:,i*2,:] = np.abs(x_tran)
        b[:,i*2+1,:] = np.arctan2(a[:,i*2,:], a[:,i*2+1,:]) / np.pi
    return b
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

# from sympy import *
# Xtrain = Mul(X_train, X_train1a)
# Xtest = Mul(X_test, X_test1a)

Xtrain = np.concatenate((X_train,X_train1a),axis=1)
Xtest = np.concatenate((X_test,X_test1a),axis=1)

in_shp = list(Xtrain.shape[1:])

from tensorflow.keras.layers import Convolution1D, Dense, Activation, Dropout, Flatten, MaxPooling2D


import tensorflow.keras.backend as K


# in_shp1=(2, 128)
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

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


x = Flatten()(x2)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(11)(x)
x_out = Activation('softmax')(x)
model = Model(inputs=ConvInput, outputs=x_out, name='CNN')
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

batch_size = 256
epochs = 30
history = model.fit(Xtrain, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(Xtest, Y_test)
                    )

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

acc_list = []
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_SNRs = list(test_SNRs)
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
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

    plot_confusion_matrix(confnorm, labels=mods, title="CNN Confusion Matrix (SNR=%d)" % (snr))
    filepath = 'C:/Users/admin/Desktop/TCN-with-attention-master/data/picture1/CNN2/' + "CNN Confusion Matrix (SNR=%d)" % (snr) + 'SNR.png'
    plt.savefig(filepath)
    plt.show()

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor

    acc[snr] = 1.0 * cor / (cor + ncor)
    print('snr:', snr)
    print('acc:', acc[snr])
    acc_list.append(acc[snr])

import pandas as pd

df1 = pd.DataFrame({'acc':acc_list})
df1.to_csv('C:/Users/admin/Desktop/TCN-with-attention-master/data/datatcn1/CNN2.csv', index=False)


plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.yticks(np.linspace(0, 1, 6))
plt.show()