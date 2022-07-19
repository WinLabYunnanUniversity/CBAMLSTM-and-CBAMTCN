
import pandas as pd
import numpy as np
#
# import matplotlib.pyplot as plt
#
# # models = ['TCN', 'DNN', 'CNN', 'LSTM', 'DetectNet', 'DetectNetTCN']
# models = ['DNN', 'LSTM', 'CNN', 'CBAMLSTM', 'CBAMTCN']
# mods = ['8PSK',  'AM-DSB', 'AM-SSB',  'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM' ]
# sample_length = [128]
#
#
# def plotResultsModel(mod, sample_length, results, pf, models):
#     plt.figure(figsize=[12, 6])
#     for i in range(len(models)):
#         plt.plot(range(-20, 20, 2), results[i], label=str(models[i]) + " pf= " + '{:.2f}%'.format(pf[i]))
#
#     plt.title(str(mod) + ' ' + str(sample_length))
#     plt.legend()
#     plt.xlim([-20, 20])
#     plt.ylim([0.0, 1.1])
#     plt.xlabel('SNR db')
#     plt.ylabel('Pd ')
#     plt.grid()
#     plt.savefig('picture1/{:s}-{:d}.png'.format(mod, sample_length))
#     plt.show()
#
#
# model = 'CBAMLSTM'
# _sample_length = 128
#
# results = []
# pf = []
# mod_names = []
# for mod in mods:
#     try:
#         print(model + '/' + str(_sample_length) + '/' + mod + 'Pds.csv')
#         doc = pd.read_csv('result/xls/' + model + '/' + str(_sample_length) + '/' + mod + 'Pds.csv', header=None)
#         line = np.array(doc)
#         results.append(line[0, range(20)])
#         pf.append(line[0, 20])
#         mod_names.append(mod)
#     except Exception as e:
#         print("Nao encontrado")
#
# plotResultsModel(model, _sample_length, results, pf, mod_names)


# _sample_length = 128
# mod = 'QAM16'
#
# results = []
# pf = []
# _models = []
# for model in models:
#     try:
#         print(model + '/' + str(_sample_length) + '/' + mod + 'Pds.csv')
#         doc = pd.read_csv('result/xls/' + model + '/' + str(_sample_length) + '/' + mod + 'Pds.csv', header=None)
#         line = np.array(doc)
#         results.append(line[0, range(20)])
#         pf.append(line[0, 20])
#         _models.append(model)
#     except Exception as e:
#         print("Nao encontrado")
#
# plotResultsModel(mod, _sample_length, results, pf, _models)


     ##画pf-pd图
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# models = ['CNN']
# mods = ['-6', '-8', '-10', '-12']
# def plotResultsModel(results,  mods):
#     plt.figure(figsize=[12, 6])
#     for i in range(len(mods)):
#         plt.plot(np.arange(0, 1, step=0.05), results[i], label=" SNR= " + str(mods[i]) + 'dB')
#
#     plt.legend()
#     x_major_locator = MultipleLocator(0.1) # 把x轴的刻度间隔设置为1，并存在变量里
#     y_major_locator = MultipleLocator(0.1)
#     ax = plt.gca()  # ax为两条坐标轴的实例
#     ax.xaxis.set_major_locator(x_major_locator)# 把x轴的主刻度设置为0.1的倍数
#     ax.yaxis.set_major_locator(y_major_locator)
#     plt.xlim([0, 1])
#     plt.ylim([0.0, 1])
#     plt.xlabel('Pf')
#     plt.ylabel('Pd')
#     plt.grid()
#     plt.savefig('pfpd/pf-pd.png')
#     plt.show()
#
# results = []
# mod_names = []
# for mod in mods:
#     try:
#         print('pfpd/' + mod + 'dB.csv')
#         doc = pd.read_csv('pfpd/' + mod + 'dB.csv', header=None)
#         line = np.array(doc)
#         results.append(line[0, range(20)])
#         mod_names.append(mod)
#     except Exception as e:
#         print("Nao encontrado")
#
# plotResultsModel(results,  mod_names)


## 画识别图
# CBAMLSTM
import matplotlib.pyplot as plt

# y1 = [0.09262025694651928,0.10051781906792567,0.10105580693815988,0.11254612546125461,0.14829123328380386,0.23144378327591605,0.36427480916030536,0.5368550368550369,
# 0.7022156573116691,0.824304100568692,0.8978300180831826,0.9031966224366706,0.9077306733167082,0.9071320182094081,0.9053398058252428,0.9076151121605667,
# 0.9120541205412054,0.9085439229843562,0.9116405307599518,0.9132824427480916]

y1 = [0.09530923214819241,0.0953396283886689,0.09321266968325792,0.11346863468634687,0.15185735512630014,0.24209207641716254,0.3719083969465649,
0.5608108108108109,0.7051698670605613,0.8302903322358576,0.8936106088004823,0.905307599517491,0.9164588528678305,0.9116843702579667,0.9108009708737864,
0.916469893742621,0.9157441574415744,0.9175691937424789,0.9164656212303981,0.9209160305343511]

# CBAMTCN
# y2 = [0.09381535703615178,0.09655802619555284,0.09351432880844646,0.1045510455104551,0.14829123328380386,0.24397118697150016,
# 0.3734351145038168,0.5730958230958231,0.744165435745938,0.8533373241544447,0.9113924050632911,0.9161640530759951,0.9217581047381546,0.9223065250379363,
# 0.9180825242718447,0.9229634002361276,0.923739237392374,0.9235860409145608,0.9243063932448733,0.9273282442748092]

y2 = [0.09411413205855991,0.09808102345415778,0.09532428355957767,0.10239852398523985,0.15572065378900446,0.26119636705292826,0.3951145038167939,0.5577395577395577,
0.7190546528803545,0.8422627955701886,0.9005424954792043,0.9134499396863691,0.9217581047381546,0.9256449165402124,0.9247572815533981,
0.9297520661157025,0.9203567035670357,0.927496991576414,0.9182750301568154,0.926412213740458]

# CNN
y3 = [0.09471168210337616,0.0953396283886689,0.09803921568627451,0.10547355473554736,0.11530460624071323,0.18321327904791732,0.3111450381679389,
0.4705159705159705,0.5946824224519941,0.7042801556420234,0.7751657625075347,0.8094089264173703,0.8204488778054863,0.8306525037936268,
0.8310072815533981,0.8376623376623377,0.8379458794587946,0.8399518652226233,0.8290108564535585,0.8354198473282443]

# LSTM
# y4 = [0.08783985658798925,0.10356381358513554,0.10105580693815988,0.11100861008610086,0.14234769687964338,0.22831193235202005,
# 0.3435114503816794,0.44594594594594594,0.5725258493353028,0.6815324753067944,0.705244122965642,0.7216525934861279,0.7325436408977556,
# 0.7411229135053111,0.7342233009708737,0.7387839433293979,0.7367773677736777,0.7490974729241877,0.753015681544029,0.7453435114503817]
y4 = [0.09799820734986556,0.09655802619555284,0.09773755656108597,0.1088560885608856,0.1447251114413076,0.2486689633573442,
0.3786259541984733,0.5224201474201474,0.6576070901033974,0.7683328344806944,0.807715491259795,0.8386610373944512,0.8494389027431422,
0.8467374810318664,0.8525485436893204,0.8485832349468713,0.8508610086100861,0.8561973525872443,0.853437876960193,0.8488549618320611]

# TCN
# y5 = [0.09232148192411115,0.1023454157782516,0.09803921568627451,0.0971709717097171,0.1298662704309064,0.23269652364547447,0.32183206106870227,
# 0.43151105651105653,0.5426883308714919,0.625860520802155,0.676913803496082,0.7104945717732207,0.7119700748129676,0.7253414264036419,
# 0.7214805825242718,0.7243211334120425,0.7164821648216482,0.733453670276775,0.7337153196622437,0.7288549618320611]
y5 = [0.09441290708096803,0.09929942126104173,0.09743589743589744,0.1045510455104551,0.12897473997028233,0.21421860319448793,
0.36335877862595417,0.5451105651105651,0.6888478581979321,0.7868751870697396,0.8266787221217601,0.8417370325693607,0.8583790523690773,
0.8540819423368741,0.8650242718446602,0.8607910271546635,0.8680688806888068,0.8635018050541517,0.868302328106152,0.8701526717557252]


x = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
plt.plot(x, y1, linewidth=1, color="blue", marker="o")
plt.plot(x, y2, linewidth=1, color="red", marker="o")
plt.plot(x, y3, linewidth=1, color="orange", marker="o")
plt.plot(x, y4, linewidth=1, color="yellow", marker="o")
plt.plot(x, y5, linewidth=1, color="green", marker="o")

plt.xlabel("SNR(dB)")
plt.ylabel("Accuracy")
plt.legend(["CBAMLSTM", "CBAMTCN","CNN","LSTM","TCN"], loc="upper left")#设置线条标识
plt.xlim([-20, 20])
plt.ylim([0.0, 1])
plt.show()