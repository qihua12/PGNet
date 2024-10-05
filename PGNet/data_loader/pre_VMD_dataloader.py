import pickle
import torch
import numpy as np
import torch.utils.data as Data
import h5py
from vmdpy import VMD


def Load_Dataset(dataset,
                 logger, data_path):
    if dataset == '2016.10a' or '2016.10c':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
    elif dataset == '2016.10b':
        classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                   'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
    elif dataset == '2018.01':
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
                    '2018.01': 'GOLD_XYZ_OSC.0001_1024.hdf5'}

    file_pointer = data_path + '/%s' % dataset_file.get(dataset)

    Signals = []
    Labels = []
    SNRs = []
    labels_and_SNR = []


    if dataset == '2016.10a' or dataset == '2016.10b' or dataset == '2016.10c':
        Set = pickle.load(open(file_pointer, 'rb'), encoding='latin1')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])

        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

                    labels_and_SNR.append((mod, snr))

        labels_and_SNR = np.array(labels_and_SNR)
        # labels_and_SNR=torch.from_numpy(labels_and_SNR.astype(np.float32))
        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))


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
        Signals = Signals.permute(0, 2, 1)  # X:(2555904, 2, 1024)

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

    return Signals, Labels, SNRs, snrs, mods, labels_and_SNR

def VMD_AMC(Signals_train, Signals_test):
    # VMD
    # some sample parameters for VMD
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 6  # 3 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7
    Signals_train_VMD = []
    Signals_test_VMD = []

    t = 0
    for signal_train in Signals_train:
        signal_train_I = list(signal_train[0, :])
        signal_train_Q = list(signal_train[1, :])
        u_I, _, _ = VMD(signal_train_I, alpha, tau, K, DC, init, tol)
        u_Q, _, _ = VMD(signal_train_Q, alpha, tau, K, DC, init, tol)
        # u_I = np.expand_dims(u_I, axis=0)
        # u_Q = np.expand_dims(u_Q, axis=0)
        # signal_train = np.expand_dims(signal_train, axis=0)
        x_con_train = np.concatenate([signal_train, u_I, u_Q], axis=0)
        # Signals_train_VMD = np.concatenate([Signals_train_VMD, x_con_train], axis=0)
        Signals_train_VMD.append(x_con_train)
        t = t + 1
        print('已完成: ', t)
    Signals_train_VMD = np.array(Signals_train_VMD)

    ts = 0
    for signal_test in Signals_test:
        signal_test_I = list(signal_test[0, :])
        signal_test_Q = list(signal_test[1, :])
        u_I_test, _, _ = VMD(signal_test_I, alpha, tau, K, DC, init, tol)
        u_Q_test, _, _ = VMD(signal_test_Q, alpha, tau, K, DC, init, tol)
        # u_I_test = np.expand_dims(u_I_test, axis=0)
        # u_Q_test = np.expand_dims(u_Q_test, axis=0)
        # signal_test = np.expand_dims(signal_test, axis=0)
        x_con_test = np.concatenate([signal_test, u_I_test, u_Q_test], axis=0)
        # Signals_test_VMD = np.concatenate([Signals_test_VMD, x_con_test], axis=0)
        Signals_test_VMD.append(x_con_test)

        ts = ts + 1
        print('已完成: ', ts)
    Signals_test_VMD = np.array(Signals_test_VMD)

    print(Signals_train_VMD.shape, Signals_test_VMD.shape)

    return Signals_train_VMD, Signals_test_VMD


def Dataset_Split(Signals,
                  Labels,
                  snrs,
                  mods,
                  logger,
                  labels_and_SNR,
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

    labels_and_SNR_train = labels_and_SNR[train_idx]
    labels_and_SNR_test = labels_and_SNR[test_idx]
    # SNRS_VMD_train = SNRS_VMD[train_idx]
    # SNRS_VMD_test = SNRS_VMD[test_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]



    '''
    # reshaping data - add one dimension
    # Signals_train = np.expand_dims(Signals_train, axis=1)
    # Signals_test = np.expand_dims(Signals_test, axis=1)
    # print(Signals_train.shape, Signals_test.shape)
    Signals_train_VMD,Signals_test_VMD=VMD_AMC(Signals_train,Signals_test)


    # reshaping data - add one dimension
    Signals_test_VMD = np.expand_dims(Signals_test_VMD, axis=1)
    Signals_train_VMD = np.expand_dims(Signals_train_VMD, axis=1)
    print(Signals_test_VMD.shape, Signals_train_VMD.shape)

    Signals_train_VMD, Signals_test_VMD = torch.Tensor(Signals_train_VMD), torch.Tensor(Signals_test_VMD)

    Signals_train = Signals_train_VMD.permute(0, 1, 3, 2)  # 换序
    Signals_test = Signals_test_VMD.permute(0, 1, 3, 2)
    print('换维', Signals_train_VMD.shape, Signals_test_VMD.shape)
    '''

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Labels_train), (Signals_test, Labels_test), (
    labels_and_SNR_train, labels_and_SNR_test), test_idx


def Dataset_Split_VMD(Signals,
                      Labels,
                      snrs,
                      mods,
                      logger,
                      labels_and_SNR,
                      test_size=0.2):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int((n_examples * (1 - test_size)) * 0.01)
    train_idx = []
    test_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int((
                                                            (n_examples - n_train) * test_size / (
                                                            (len(mods) * len(snrs)) * test_size)) * 0.1),
                                           replace=False)

        train_idx = np.hstack([train_idx, train_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    train_idx = train_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    labels_and_SNR_train = labels_and_SNR[train_idx]
    labels_and_SNR_test = labels_and_SNR[test_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    '''
    # reshaping data - add one dimension
    # Signals_train = np.expand_dims(Signals_train, axis=1)
    # Signals_test = np.expand_dims(Signals_test, axis=1)
    # print(Signals_train.shape, Signals_test.shape)
    Signals_train_VMD,Signals_test_VMD=VMD_AMC(Signals_train,Signals_test)


    # reshaping data - add one dimension
    Signals_test_VMD = np.expand_dims(Signals_test_VMD, axis=1)
    Signals_train_VMD = np.expand_dims(Signals_train_VMD, axis=1)
    print(Signals_test_VMD.shape, Signals_train_VMD.shape)

    Signals_train_VMD, Signals_test_VMD = torch.Tensor(Signals_train_VMD), torch.Tensor(Signals_test_VMD)

    Signals_train = Signals_train_VMD.permute(0, 1, 3, 2)  # 换序
    Signals_test = Signals_test_VMD.permute(0, 1, 3, 2)
    print('换维', Signals_train_VMD.shape, Signals_test_VMD.shape)
    '''

    logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
    logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
    logger.info('*' * 20)

    return (Signals_train, Labels_train), (Signals_test, Labels_test), (
    labels_and_SNR_train, labels_and_SNR_test), test_idx


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
