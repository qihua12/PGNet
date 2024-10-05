import pickle
import torch
import numpy as np
import torch.utils.data as Data
import h5py
import scipy.io as scio

# from gene_fun_quan import Gene


def D_2_128_Load_Dataset(dataset,
                         logger, data_path):
    if dataset == '2016.10a' or '2016.10c':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
    elif dataset == '2016.10b' or '2022.01a':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
    elif dataset == '2018.01a':
        classes = {b'00K': 0, b'4ASK': 1, b'8ASK': 2, b'BPSK': 3, b'QPSK': 4,
                   b'8PSK': 5, b'16PSK': 6, b'32PSK': 7, b'16APSK': 8, b'32APSK': 9,
                   b'64APSK': 10, b'128APSK': 11, b'16QAM': 12, b'32QAM': 13, b'64QAM': 14,
                   b'128QAM': 15, b'256QAM': 16, b'AM-SSB-WC': 17, b'AM-SSB-SC': 18,
                   b'AM-DSB-WC': 19, b'AM-DSB-SC': 20, b'FM': 21, b'GMSK': 22, b'OQPSK': 23}
    else:
        raise NotImplementedError(f'Not Implemented dataset:{dataset}')

    dataset_file = {'2016.10a': 'RML2016.10a_dict.pkl',
                    '2016.10b': 'RML2016.10b.dat',
                    '2016.10c': '2016.04C.multisnr.pkl',
                    '2018.01a': 'GOLD_XYZ_OSC.0001_1024.hdf5',
                    '2022.01a': 'RML22.pickle.01A',
                    }

    file_pointer = data_path + '/%s' % dataset_file.get(dataset)

    Signals = []
    Labels = []
    SNRs = []

    if dataset == '2016.10a' or dataset == '2016.10b' or dataset == '2016.10c' or dataset == '2022.01a':
        Set = pickle.load(open(file_pointer, 'rb'), encoding='latin1')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])

        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))
        # Signals = Signals.permute(0, 1, 2)  # 变换数据1,2,128
        # Signals = Signals.permute(0, 2, 1)#1,128,2

        Labels = [classes[i] for i in Labels]  # mapping modulation formats(str) to int
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

    else:
        f = h5py.File(file_pointer)
        Signals = f['X'][:]
        Labels = f['Y'][:]
        SNRs = f['Z'][:]
        f.close()

        Signals = torch.from_numpy(Signals.astype(np.float32))
        # Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)

        SNRs = SNRs.tolist()
        snrs = list(np.unique(SNRs))
        mods = list(classes.keys())

        Labels = np.argwhere(Labels == 1)[:, 1]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

    logger.info('*' * 20)
    logger.info(f'Signals.shape: {list(Signals.shape)}')
    logger.info(f'Labels.shape: {list(Labels.shape)}')
    logger.info('*' * 20)

    return Signals, Labels, SNRs, snrs, mods


def Load_Dataset(dataset,
                 logger, data_path):
    if dataset == '2016.10a' or '2016.10c':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
    elif dataset == '2016.10b' or '2022.01a':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
    elif dataset == '2018.01a':
        classes = {b'00K': 0, b'4ASK': 1, b'8ASK': 2, b'BPSK': 3, b'QPSK': 4,
                   b'8PSK': 5, b'16PSK': 6, b'32PSK': 7, b'16APSK': 8, b'32APSK': 9,
                   b'64APSK': 10, b'128APSK': 11, b'16QAM': 12, b'32QAM': 13, b'64QAM': 14,
                   b'128QAM': 15, b'256QAM': 16, b'AM-SSB-WC': 17, b'AM-SSB-SC': 18,
                   b'AM-DSB-WC': 19, b'AM-DSB-SC': 20, b'FM': 21, b'GMSK': 22, b'OQPSK': 23}
    else:
        raise NotImplementedError(f'Not Implemented dataset:{dataset}')

    dataset_file = {'2016.10a': 'RML2016.10a_dict.pkl',
                    '2016.10b': 'RML2016.10b.dat',
                    '2016.10c': '2016.04C.multisnr.pkl',
                    '2018.01a': 'GOLD_XYZ_OSC.0001_1024.hdf5',
                    '2022.01a': 'RML22.pickle.01A',
                    }

    file_pointer = data_path + '/%s' % dataset_file.get(dataset)

    Signals = []
    Labels = []
    SNRs = []

    if dataset == '2016.10a' or dataset == '2016.10b' or dataset == '2016.10c' or dataset == '2022.01a':
        Set = pickle.load(open(file_pointer, 'rb'), encoding='latin1')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])

        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))
        # Signals = Signals.permute(0, 1, 2)  # 变换数据1,2,128
        Signals = Signals.permute(0, 2, 1)  # 1,128,2

        Labels = [classes[i] for i in Labels]  # mapping modulation formats(str) to int
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

    else:
        f = h5py.File(file_pointer)
        Signals = f['X'][:]
        Labels = f['Y'][:]
        SNRs = f['Z'][:]
        f.close()

        Signals = torch.from_numpy(Signals.astype(np.float32))
        # Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)

        SNRs = SNRs.tolist()
        snrs = list(np.unique(SNRs))
        mods = list(classes.keys())

        Labels = np.argwhere(Labels == 1)[:, 1]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

    logger.info('*' * 20)
    logger.info(f'Signals.shape: {list(Signals.shape)}')
    logger.info(f'Labels.shape: {list(Labels.shape)}')
    logger.info('*' * 20)

    return Signals, Labels, SNRs, snrs, mods


def Duo_Load_Dataset(dataset,
                     logger, data_path):
    if dataset == '2016.10a' or '2016.10c':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}

    elif dataset == '2016.10b' or '2022.01a':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
    elif dataset == '2018.01a':
        classes = {b'00K': 0, b'4ASK': 1, b'8ASK': 2, b'BPSK': 3, b'QPSK': 4,
                   b'8PSK': 5, b'16PSK': 6, b'32PSK': 7, b'16APSK': 8, b'32APSK': 9,
                   b'64APSK': 10, b'128APSK': 11, b'16QAM': 12, b'32QAM': 13, b'64QAM': 14,
                   b'128QAM': 15, b'256QAM': 16, b'AM-SSB-WC': 17, b'AM-SSB-SC': 18,
                   b'AM-DSB-WC': 19, b'AM-DSB-SC': 20, b'FM': 21, b'GMSK': 22, b'OQPSK': 23}
    else:
        raise NotImplementedError(f'Not Implemented dataset:{dataset}')

    dataset_file = {'2016.10a': 'RML2016.10a_dict.pkl',
                    '2016.10b': 'RML2016.10b.dat',
                    '2016.10c': '2016.04C.multisnr.pkl',
                    '2018.01a': 'GOLD_XYZ_OSC.0001_1024.hdf5',
                    '2022.01a': 'RML22.pickle.01A',
                    }

    file_pointer = data_path + '/%s' % dataset_file.get(dataset)

    Signals = []
    Labels = []
    SNRs = []

    if dataset == '2016.10a' or dataset == '2016.10b' or dataset == '2016.10c' or dataset == '2022.01a':
        Set = pickle.load(open(file_pointer, 'rb'), encoding='latin1')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])

        for mod in mods:
            for snr in snrs:
                # if snr > 16:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))
        Signals = Signals.permute(0, 1, 2)  # 变换数据1,2,128
        # Signals = Signals.permute(0, 2, 1)#1,128,2
        # Signals_real = Signals[:, 0, :]
        # Signals_imag = Signals[:, 1, :]
        # Signals = torch.complex(Signals_real, Signals_imag)

        Labels = [classes[i] for i in Labels]  # mapping modulation formats(str) to int
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)
        #
        # gene_func = Gene(Signals, Labels)
        # #
        # # feature0 = gene_func.statistic()  # 统计特征
        # feature1 = gene_func.Freq()  # 频域特征
        # feature2 = gene_func.Time() #时域特征
        # feature3 = gene_func.calculate_wavelet_packet() #小波包分解
        # feature4 = gene_func.cwt() #计算信号的连续小波变换（CWT）
        #
        # train = np.concatenate((feature1, feature2, feature3, feature4), axis=1)
        # real_part = np.expand_dims(np.real(train), axis=1)
        # imag_part = np.expand_dims(np.imag(train), axis=1)
        # train = np.concatenate((real_part, imag_part), axis=1)
        # Signals = np.concatenate((Signals, train), axis=2)



    else:
        f = h5py.File(file_pointer)
        Signals = f['X'][:]
        Labels = f['Y'][:]
        SNRs = f['Z'][:]
        f.close()

        Signals = torch.from_numpy(Signals.astype(np.float32))
        # Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)

        SNRs = SNRs.tolist()
        snrs = list(np.unique(SNRs))
        mods = list(classes.keys())

        Labels = np.argwhere(Labels == 1)[:, 1]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

    logger.info('*' * 20)
    logger.info(f'Signals.shape: {list(Signals.shape)}')
    logger.info(f'Labels.shape: {list(Labels.shape)}')
    logger.info('*' * 20)

    return Signals, Labels, SNRs, snrs, mods


def C_Dataset_Split(Signals,
                    Labels,
                    snrs,
                    mods,
                    logger,
                    test_size=0.2):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - test_size))

    train_idx = []
    test_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int(
                                               (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * test_size)),
                                           replace=False)

        train_idx = np.hstack([train_idx, train_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    train_idx = train_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]
    # print(Labels_test)
    # reshaping data - add one dimension
    Signals_train = torch.unsqueeze(Signals_train, 1)
    Signals_test = torch.unsqueeze(Signals_test, 1)
    # 复数的添加两个维度
    # Signals_train = torch.unsqueeze(Signals_train,-1)
    # Signals_test = torch.unsqueeze(Signals_test,-1)
    print(Signals_train.shape, Signals_test.shape)

    # Signals_train = Signals_train.permute(0, 1, 3, 2)  # 换序
    # Signals_test = Signals_test.permute(0, 1, 3, 2)
    # print('换维',Signals_train.shape, Signals_test.shape)

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Labels_train), (Signals_test, Labels_test), test_idx


def D_2_Dataset_Split(Signals,
                      Labels,
                      snrs,
                      mods,
                      logger,
                      test_size=0.2):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - test_size))

    train_idx = []
    test_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int(
                                               (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * test_size)),
                                           replace=False)

        train_idx = np.hstack([train_idx, train_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    ##部分选取数据集

    train_idx = train_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    # reshaping data - add one dimension

    print(Signals_train.shape, Signals_test.shape)

    Signals_train, Signals_test = torch.Tensor(Signals_train), torch.Tensor(Signals_test)

    # Signals_train = Signals_train.permute(0, 1, 3, 2)  # 换序
    # Signals_test = Signals_test.permute(0, 1, 3, 2)
    # print('换维',Signals_train.shape, Signals_test.shape)

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Labels_train), (Signals_test, Labels_test), test_idx


def MCLDNN_Dataset_Split(Signals,
                         Labels,
                         snrs,
                         mods,
                         logger,
                         test_size=0.2):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - test_size))

    train_idx = []
    test_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int(
                                               (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * test_size)),
                                           replace=False)

        train_idx = np.hstack([train_idx, train_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    ##部分选取数据集

    train_idx = train_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    Signals_test2 = Signals_test[:, 0, :]
    Signals_test3 = Signals_test[:, 1, :]
    # reshaping data - add one dimension

    Signals_train2 = Signals_train[:, 0, :]
    Signals_train3 = Signals_train[:, 1, :]

    Signals_train = np.expand_dims(Signals_train, axis=1)
    Signals_test = np.expand_dims(Signals_test, axis=1)

    Signals_train2 = np.expand_dims(Signals_train2, axis=1)
    Signals_train3 = np.expand_dims(Signals_train3, axis=1)

    Signals_test2 = np.expand_dims(Signals_test2, axis=1)
    Signals_test3 = np.expand_dims(Signals_test3, axis=1)


    print(Signals_train.shape, Signals_train2.shape, Signals_train3.shape, Signals_test.shape, Signals_test2.shape,
          Signals_test3.shape)

    Signals_train, Signals_train2, Signals_train3, Signals_test, Signals_test2, Signals_test3 = torch.Tensor(
        Signals_train), \
                                                                                                torch.Tensor(
                                                                                                    Signals_train2), \
                                                                                                torch.Tensor(
                                                                                                    Signals_train3), \
                                                                                                torch.Tensor(
                                                                                                    Signals_test), \
                                                                                                torch.Tensor(
                                                                                                    Signals_test2), \
                                                                                                torch.Tensor(
                                                                                                    Signals_test3)

    # Signals_train = Signals_train.permute(0, 1, 3, 2)  # 换序
    # Signals_test = Signals_test.permute(0, 1, 3, 2)
    # print('换维',Signals_train.shape, Signals_test.shape)

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Signals_train2, Signals_train3, Labels_train), (
        Signals_test, Signals_test2, Signals_test3, Labels_test), test_idx


def Dataset_Split(Signals,
                  Labels,
                  snrs,
                  mods,
                  logger,
                  test_size=0.2):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - test_size))

    train_idx = []
    test_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int(
                                               (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * test_size)),
                                           replace=False)

        train_idx = np.hstack([train_idx, train_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    ##部分选取数据集

    train_idx = train_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    # reshaping data - add one dimension
    Signals_train = np.expand_dims(Signals_train, axis=1)
    Signals_test = np.expand_dims(Signals_test, axis=1)

    print(Signals_train.shape, Signals_test.shape)

    Signals_train, Signals_test = torch.Tensor(Signals_train), torch.Tensor(Signals_test)

    # Signals_train = Signals_train.permute(0, 1, 3, 2)  # 换序
    # Signals_test = Signals_test.permute(0, 1, 3, 2)
    # print('换维',Signals_train.shape, Signals_test.shape)

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Labels_train), (Signals_test, Labels_test), test_idx


def Small_Dataset_Split(Signals,
                        Labels,
                        snrs,
                        mods,
                        logger,
                        test_size=0.5):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - test_size))

    train_idx = []
    test_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int((n_train / (len(mods) * len(snrs))) * 0.05),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int((
                                                            (n_examples - n_train) * test_size / (
                                                            (len(mods) * len(snrs)) * test_size)) * 0.05),
                                           replace=False)

        train_idx = np.hstack([train_idx, train_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    train_idx = train_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    # reshaping data - add one dimension
    Signals_train = np.expand_dims(Signals_train, axis=1)
    Signals_test = np.expand_dims(Signals_test, axis=1)

    print(Signals_train.shape, Signals_test.shape)

    Signals_train, Signals_test = torch.Tensor(Signals_train), torch.Tensor(Signals_test)

    # Signals_train = Signals_train.permute(0, 1, 3, 2)  # 换序
    # Signals_test = Signals_test.permute(0, 1, 3, 2)
    # print('换维',Signals_train.shape, Signals_test.shape)

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Labels_train), (Signals_test, Labels_test), test_idx


def duo_Dataset_Split(Signals,
                      Labels,
                      snrs,
                      mods,
                      logger,
                      test_size=0.2):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - test_size))

    train_idx = []
    test_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int(
                                               (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * test_size)),
                                           replace=False)

        train_idx = np.hstack([train_idx, train_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    train_idx = train_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    #
    # 导入特征。
    dict = scio.loadmat('data.mat')  # 输出的为dict字典类型
    feature0 = dict['feature0']
    feature1 = dict['feature1']
    feature2 = dict['feature2']
    feature3 = dict['feature3']
    feature4 = dict['feature4']

    # 特征分组
    # feature0 = np.concatenate((feature0, feature1), axis=1)
    feature0_train = torch.Tensor(feature0[train_idx])  # 统计特征
    feature1_train = torch.Tensor(feature1[train_idx])  # 频域特征
    feature2_train = torch.Tensor(feature2[train_idx])  # 时域特征
    feature3_train = torch.Tensor(feature3[train_idx])  # 小波包分解
    feature4_train = torch.Tensor(feature4[train_idx])  # 小波变换

    feature0_test = torch.Tensor(feature0[test_idx])
    feature1_test = torch.Tensor(feature1[test_idx])
    feature2_test = torch.Tensor(feature2[test_idx])
    feature3_test = torch.Tensor(feature3[test_idx])
    feature4_test = torch.Tensor(feature4[test_idx])

    # 输入特征
    # 统计特征不要了
    # feature0_train = feature0_train[:, 4:10]  # 4个
    # feature0_test = feature0_test[:, 4:10]
    # 频域特征功率谱均值，功率谱方差，功率谱熵，功率谱重心
    feature1_train = feature1_train[:, 2:9]  # 4个
    feature1_test = feature1_test[:, 2:9]
    #
    # 时域特征为峰值因子，脉冲因子，裕度因子
    # feature2_train = feature2_train[:, 5:8]  # 3个
    # input_train = np.expand_dims(input_train, axis=1)
    # feature2_test = feature2_test[:, 5:8]
    # input_test = np.expand_dims(input_test, axis=1)

    # input_train = torch.Tensor(input_train)
    # input_test = torch.Tensor(input_test)

    FB_train = np.concatenate((feature1_train, feature3_train, feature4_train),
                              axis=1)
    FB_test = np.concatenate((feature1_test, feature3_test, feature4_test), axis=1)

    real_part_train = np.expand_dims(np.real(FB_train), axis=1)
    imag_part_train = np.expand_dims(np.imag(FB_train), axis=1)
    FB_data_train = np.concatenate((real_part_train, imag_part_train), axis=1)

    real_part_test = np.expand_dims(np.real(FB_test), axis=1)
    imag_part_test = np.expand_dims(np.imag(FB_test), axis=1)
    FB_data_test = np.concatenate((real_part_test, imag_part_test), axis=1)

    # Signals_train = np.concatenate((Signals_train, input_data_train), axis=2)
    # Signals_test = np.concatenate((Signals_test, input_data_test), axis=2)

    # reshaping data - add one dimension
    FB_data_train = np.expand_dims(FB_data_train, axis=1)
    FB_data_test = np.expand_dims(FB_data_test, axis=1)

    Signals_train = np.expand_dims(Signals_train, axis=1)
    Signals_test = np.expand_dims(Signals_test, axis=1)

    print(FB_data_train.shape, FB_data_test.shape)
    print(Signals_train.shape, Signals_test.shape)

    Signals_train, Signals_test = torch.Tensor(Signals_train), torch.Tensor(Signals_test)
    FB_data_train, FB_data_test = torch.Tensor(FB_data_train), torch.Tensor(FB_data_test)

    # Signals_train = Signals_train.permute(0, 1, 3, 2)  # 换序
    # Signals_test = Signals_test.permute(0, 1, 3, 2)
    # print('换维',Signals_train.shape, Signals_test.shape)

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, FB_data_train, Labels_train), (Signals_test, FB_data_test, Labels_test), train_idx, test_idx


def Create_Data_Loader(train_set, val_set, cfg, logger):
    train_data = Data.TensorDataset(*train_set)
    val_data = Data.TensorDataset(*val_set)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    logger.info(f"train_loader batch: {len(train_loader)}")
    logger.info(f"val_loader batch: {len(val_loader)}")

    return train_loader, val_loader
