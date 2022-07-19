import utils
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint, TensorBoard

# from sendMail import alertTrainEnded

# # GPU usage setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for sample_length in [128]:
    for mod_name in ['signal']:

        # hyperparameters
        lr = 0.0001
        drop_ratio = 0.2

        max_epoch = 3
        batch_size = 200
        patience = 6

        # sample_length = 128
        swap_dim = False
        if (swap_dim):
            input_dim = (sample_length, 2)
        else:
            input_dim = (2, sample_length)

        # load data
        # 数据预处理
        filename = './RML2016.10a_dict.pkl'
        x_train, y_train, x_val, y_val, x_test, y_test, val_SNRs, test_SNRs = utils.radioml_IQ_dataH1H0(filename, mod_name,
                                                                                                    swap_dim=swap_dim)

        # callbacks
        # 在第一段，采用6个周期的patience来停止训练模型收敛
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        best_model_path = 'result/models/DNN1/' + str(sample_length) + '/' + str(mod_name) + 'best.h5'
        checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True)
        TB_dir = 'result/TB/DNN1/' + str(mod_name) + '_' + str(sample_length)
        tensorboard = TensorBoard(TB_dir)

        # 调用该网络模型
        # 通过该网络模型训练
        model = utils.DNN(lr, input_dim, drop_ratio)
        history = model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=1, shuffle=True,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping, checkpointer, tensorboard])
        print('Fisrt stage finished, loss is stable')

        pf_min = 0.0    #pf停止时间间隔最小值
        pf_max = 10.0   #pf停止时间间隔最大值
        pf_test = LambdaCallback(
            on_epoch_end=lambda epoch,
                                logs: utils.get_pf(x_val, y_val, val_SNRs, model, epoch, pf_min, pf_max))

        print('Start second stage, trade-off metrics')
        model = load_model(best_model_path)
        model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=1, shuffle=True,
                  callbacks=[pf_test])

        # pf落入时间间隔时停止训练
        if model.stop_training:
            # save results
            model.save('result/models/DNN1/' + str(sample_length) + '/' + str(mod_name) + 'final.h5')
            print('Second stage finished, get the final model')
            save_path = 'C:/Users/admin/Desktop/TCN-with-attention-master/data/result/xls/DNN1/128/' + str(
                mod_name) + 'Pds.csv'
            utils.performance_evaluationH1H0(save_path, x_test, y_test, test_SNRs, model)
        else:
            print("Can't meet pf lower bound")

