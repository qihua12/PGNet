import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from util.visualize import Draw_Confmat, Snr_Acc_Plot
from sklearn.manifold import TSNE
from time import time
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns

classestsen = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK',
               'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK', 'AM-SSB']


def MCLDNN_Run_Eval(model,
                    test_set,
                    SNRs,
                    test_idx,
                    output_DIR,
                    cfg,
                    logger):
    global cm
    model.eval()
    with torch.no_grad():
        snrs = list(np.unique(SNRs))
        mods = list(cfg.classes.keys())

        Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
        Accuracy_list = np.zeros(len(snrs), dtype=float)

        # all 混淆矩阵
        sig_test, sig_test2, sig_test3, lab_test = test_set
        Sample = torch.chunk(sig_test, cfg.test_batch_size, dim=0)
        Sample2 = torch.chunk(sig_test2, cfg.test_batch_size, dim=0)
        Sample3 = torch.chunk(sig_test3, cfg.test_batch_size, dim=0)
        Label = torch.chunk(lab_test, cfg.test_batch_size, dim=0)
        pred_all = []
        label_all = []
        for (sample, sample2, sample3, label) in zip(Sample, Sample2, Sample3, Label):
            sample = sample.to(cfg.device)
            sample2 = sample2.to(cfg.device)
            sample3 = sample3.to(cfg.device)
            logit, tsne = model(sample, sample2, sample3)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_all.append(pre_lab)
            label_all.append(label)
        pred_all = np.concatenate(pred_all)
        label_all = np.concatenate(label_all)
        plt.figure(figsize=(10, 10))
        cm = confusion_matrix(label_all, pred_all, normalize='pred')
        plot_confusion_matrix(cm, labels=mods)

        filepath = output_DIR + "confusion matrix" + '.png'
        plt.savefig(filepath, format='png', dpi=600)
        plt.close()
        # Classification Report
        report = classification_report(label_all, pred_all, target_names=mods)
        print(report)

        pre_lab_all = []
        label_all = []
        acc = {}
        for snr_i, snr in enumerate(snrs):
            test_SNRs = map(lambda x: SNRs[x], test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
            test_sig_i_2 = sig_test2[np.where(np.array(test_SNRs) == snr)]
            test_sig_i_3 = sig_test3[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
            Sample2 = torch.chunk(test_sig_i_2, cfg.test_batch_size, dim=0)
            Sample3 = torch.chunk(test_sig_i_3, cfg.test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)
            pred_i = []
            label_i = []
            feature_map_list = []
            labels = []
            for (sample, sample2, sample3, label) in zip(Sample, Sample2, Sample3, Label):
                sample = sample.to(cfg.device)
                sample2 = sample3.to(cfg.device)
                sample3 = sample3.to(cfg.device)
                logit, tsne = model(sample, sample2, sample3)
                pre_lab = torch.argmax(logit, 1).cpu()
                pred_i.append(pre_lab)
                label_i.append(label)
                for i in label.tolist():
                    tag = classestsen[i]
                    labels.append(tag)
                feature_map_list.append(tsne.cpu().detach().numpy())
            feature_map_list = np.concatenate(feature_map_list)
            t_sne(np.array(feature_map_list), np.array(labels), snr, output_DIR)
            pred_i = np.concatenate(pred_i)
            label_i = np.concatenate(label_i)

            pre_lab_all.append(pred_i)
            label_all.append(label_i)

            Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
            cm = confusion_matrix(label_i, pred_i)
            confnorm = np.zeros([len(mods), len(mods)])
            for i in range(0, len(mods)):
                confnorm[i, :] = cm[i, :] / np.sum(cm[i, :])
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(confnorm, labels=mods, title="Confusion Matrix (SNR=%d)" % (snr))
            filepath = output_DIR + "Confusion Matrix (SNR=%d)" % snr + 'SNR.png'
            plt.savefig(filepath, format='png', dpi=600)
            plt.close()
            cor = np.sum(np.diag(cm))
            ncor = np.sum(cm) - cor
            print()
            print("SNR:{} Accuracy:{} ".format(snr, cor / (cor + ncor)))
            acc[snr] = 1.0 * cor / (cor + ncor)
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        logger.info(f'overall accuracy is: {acc}')
        logger.info(f'macro F1-score is: {F1_score}')
        logger.info(f'kappa coefficient is: {kappa}')

        if cfg.Draw_Acc_Curve is True:
            Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)
        if cfg.Draw_Confmat is True:
            Draw_Confmat(Confmat_Set, snrs, cfg)


def t_sne(data, label, snr, output_DIR,dataset_name):
    # t-sne处理
    # print('starting T-SNE process')
    # start_time = time()
    global custom_palette
    data = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
    df.insert(loc=1, column='label', value=label)
    # end_time = time()
    # print('Finished')
    # 绘图
    # plt.figure(figsize=(8, 8))
    # 自定义颜色调色板
    if dataset_name == '2016.10a' or dataset_name == '2016.10c':
        custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF']
    elif dataset_name == '2016.10b' or dataset_name == '2022.01a':
        custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    else:
        raise NotImplementedError(f'Not Implemented dataset:{dataset_name}')

    # plt.figure(figsize=(8, 8))
    sns.scatterplot(x='x', y='y', hue='label', size=10, palette=custom_palette, data=df)  # 绘制散点图 palette调色板
    set_plt()

    filepathtsne = output_DIR + "tsne (SNR=%d)" % snr + 'SNR.png'
    plt.savefig(filepathtsne, format='png', dpi=600)
    # plt.show()
    plt.close()


def set_plt():
    plt.legend(title='')
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks([])


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues, labels=None):
    if labels is None:
        labels = []
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar(ticks = np.arange(0.0,1.,.2))
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    iters = np.reshape([[[i, j] for j in range(len(labels))] for i in range(len(labels))], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, round(cm[i, j], 2), va='center', ha='center', fontsize=18)  # 显示对应的数字

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')


def Small_plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues, labels=None):
    if labels is None:
        labels = []
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar(ticks = np.arange(0.0,1.,.2))
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    iters = np.reshape([[[i, j] for j in range(11)] for i in range(11)], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, round(cm[i, j], 2), va='center', ha='center', fontsize=18)  # 显示对应的数字

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')


def Small_Run_Eval(model,
                   sig_test,
                   lab_test,
                   SNRs,
                   test_idx,
                   output_DIR,
                   cfg,
                   logger):
    global cm
    model.eval()
    with torch.no_grad():
        snrs = list(np.unique(SNRs))
        mods = list(classestsen)

        Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
        Accuracy_list = np.zeros(len(snrs), dtype=float)

        # all 混淆矩阵
        Sample = torch.chunk(sig_test, cfg.test_batch_size, dim=0)
        Label = torch.chunk(lab_test, cfg.test_batch_size, dim=0)
        pred_all = []
        label_all = []
        for (sample, label) in zip(Sample, Label):
            sample = sample.to(cfg.device)
            logit, tsne = model(sample)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_all.append(pre_lab)
            label_all.append(label)
        pred_all = np.concatenate(pred_all)
        label_all = np.concatenate(label_all)
        plt.figure(figsize=(10, 10))
        cm = confusion_matrix(label_all, pred_all, normalize='pred')
        plot_confusion_matrix(cm, labels=mods)

        filepath = output_DIR + "confusion matrix" + '.png'
        plt.savefig(filepath, format='png', dpi=600)
        plt.close()
        # Classification Report
        report = classification_report(label_all, pred_all, target_names=mods)
        print(report)

        pre_lab_all = []
        label_all = []
        acc = {}
        for snr_i, snr in enumerate(snrs):
            test_SNRs = map(lambda x: SNRs[x], test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)
            pred_i = []
            label_i = []
            feature_map_list = []
            labels = []
            for (sample, label) in zip(Sample, Label):
                sample = sample.to(cfg.device)
                logit, tsne = model(sample)
                pre_lab = torch.argmax(logit, 1).cpu()
                pred_i.append(pre_lab)
                label_i.append(label)
                for i in label.tolist():
                    tag = classestsen[i]
                    labels.append(tag)
                feature_map_list.append(tsne.cpu().detach().numpy())
            feature_map_list = np.concatenate(feature_map_list)
            t_sne(np.array(feature_map_list), np.array(labels), snr, output_DIR)
            pred_i = np.concatenate(pred_i)
            label_i = np.concatenate(label_i)

            pre_lab_all.append(pred_i)
            label_all.append(label_i)

            Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
            cm = confusion_matrix(label_i, pred_i)
            confnorm = np.zeros([len(mods), len(mods)])
            for i in range(0, len(mods)):
                confnorm[i, :] = cm[i, :] / np.sum(cm[i, :])
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(confnorm, labels=mods, title="Confusion Matrix (SNR=%d)" % (snr))
            filepath = output_DIR + "Confusion Matrix (SNR=%d)" % snr + 'SNR.png'
            plt.savefig(filepath, format='png', dpi=600)
            plt.close()
            cor = np.sum(np.diag(cm))
            ncor = np.sum(cm) - cor
            print()
            print("SNR:{} Accuracy:{} ".format(snr, cor / (cor + ncor)))
            acc[snr] = 1.0 * cor / (cor + ncor)
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        logger.info(f'overall accuracy is: {acc}')
        logger.info(f'macro F1-score is: {F1_score}')
        logger.info(f'kappa coefficient is: {kappa}')

        if cfg.Draw_Acc_Curve is True:
            Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)
        if cfg.Draw_Confmat is True:
            Draw_Confmat(Confmat_Set, snrs, cfg)


def Run_Eval(model,
             sig_test,
             lab_test,
             SNRs,
             test_idx,
             output_DIR,
             cfg,
             logger):
    global cm
    model.eval()
    with torch.no_grad():
        snrs = list(np.unique(SNRs))
        mods = list(cfg.classes.keys())

        Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
        Accuracy_list = np.zeros(len(snrs), dtype=float)

        # all 混淆矩阵
        Sample = torch.chunk(sig_test, cfg.test_batch_size, dim=0)
        Label = torch.chunk(lab_test, cfg.test_batch_size, dim=0)
        pred_all = []
        label_all = []
        for (sample, label) in zip(Sample, Label):
            sample = sample.to(cfg.device)
            logit = model(sample)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_all.append(pre_lab)
            label_all.append(label)
        pred_all = np.concatenate(pred_all)
        label_all = np.concatenate(label_all)
        plt.figure(figsize=(10, 10))
        cm = confusion_matrix(label_all, pred_all, normalize='pred')
        plot_confusion_matrix(cm, labels=mods)

        filepath = output_DIR + "confusion matrix" + '.png'
        plt.savefig(filepath, format='png', dpi=600)
        plt.close()
        # Classification Report
        report = classification_report(label_all, pred_all, target_names=mods)
        print(report)

        pre_lab_all = []
        label_all = []
        acc = {}
        for snr_i, snr in enumerate(snrs):
            test_SNRs = map(lambda x: SNRs[x], test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)
            pred_i = []
            label_i = []
            feature_map_list = []
            labels = []
            for (sample, label) in zip(Sample, Label):
                sample = sample.to(cfg.device)
                logit = model(sample)
                pre_lab = torch.argmax(logit, 1).cpu()
                pred_i.append(pre_lab)
                label_i.append(label)
                for i in label.tolist():
                    tag = classestsen[i]
                    labels.append(tag)

            pred_i = np.concatenate(pred_i)
            label_i = np.concatenate(label_i)

            pre_lab_all.append(pred_i)
            label_all.append(label_i)

            Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
            cm = confusion_matrix(label_i, pred_i)
            confnorm = np.zeros([len(mods), len(mods)])
            for i in range(0, len(mods)):
                confnorm[i, :] = cm[i, :] / np.sum(cm[i, :])
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(confnorm, labels=mods, title="Confusion Matrix (SNR=%d)" % (snr))
            filepath = output_DIR + "Confusion Matrix (SNR=%d)" % snr + 'SNR.png'
            plt.savefig(filepath, format='png', dpi=600)
            plt.close()
            cor = np.sum(np.diag(cm))
            ncor = np.sum(cm) - cor
            print()
            print("SNR:{} Accuracy:{} ".format(snr, cor / (cor + ncor)))
            acc[snr] = 1.0 * cor / (cor + ncor)
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        logger.info(f'overall accuracy is: {acc}')
        logger.info(f'macro F1-score is: {F1_score}')
        logger.info(f'kappa coefficient is: {kappa}')

        if cfg.Draw_Acc_Curve is True:
            Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)
        if cfg.Draw_Confmat is True:
            Draw_Confmat(Confmat_Set, snrs, cfg)


def tsen_Run_Eval(model,
                  sig_test,
                  lab_test,
                  SNRs,
                  test_idx,
                  args,
                  cfg,
                  logger):
    global cm
    output_DIR = args.output_DIR
    dataset_name = args.dataset
    model.eval()
    with torch.no_grad():
        snrs = list(np.unique(SNRs))
        mods = list(cfg.classes.keys())

        Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
        Accuracy_list = np.zeros(len(snrs), dtype=float)

        # all 混淆矩阵
        Sample = torch.chunk(sig_test, cfg.test_batch_size, dim=0)
        Label = torch.chunk(lab_test, cfg.test_batch_size, dim=0)
        pred_all = []
        label_all = []
        for (sample, label) in zip(Sample, Label):
            sample = sample.to(cfg.device)
            logit, tsne = model(sample)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_all.append(pre_lab)
            label_all.append(label)
        pred_all = np.concatenate(pred_all)
        label_all = np.concatenate(label_all)
        plt.figure(figsize=(10, 10))
        cm = confusion_matrix(label_all, pred_all, normalize='pred')
        plot_confusion_matrix(cm, labels=mods)

        filepath = output_DIR + "confusion matrix" + '.png'
        plt.savefig(filepath, format='png', dpi=600)
        plt.close()
        # Classification Report
        report = classification_report(label_all, pred_all, target_names=mods)
        print(report)

        pre_lab_all = []
        label_all = []
        acc = {}
        for snr_i, snr in enumerate(snrs):
            test_SNRs = map(lambda x: SNRs[x], test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)
            pred_i = []
            label_i = []
            feature_map_list = []
            labels = []
            for (sample, label) in zip(Sample, Label):
                sample = sample.to(cfg.device)
                logit, tsne = model(sample)
                pre_lab = torch.argmax(logit, 1).cpu()
                pred_i.append(pre_lab)
                label_i.append(label)
                for i in label.tolist():
                    tag = classestsen[i]
                    labels.append(tag)
                feature_map_list.append(tsne.cpu().detach().numpy())
            feature_map_list = np.concatenate(feature_map_list)
            t_sne(np.array(feature_map_list), np.array(labels), snr, output_DIR,dataset_name)
            pred_i = np.concatenate(pred_i)
            label_i = np.concatenate(label_i)

            pre_lab_all.append(pred_i)
            label_all.append(label_i)

            Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
            cm = confusion_matrix(label_i, pred_i)
            confnorm = np.zeros([len(mods), len(mods)])
            for i in range(0, len(mods)):
                confnorm[i, :] = cm[i, :] / np.sum(cm[i, :])
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(confnorm, labels=mods, title="Confusion Matrix (SNR=%d)" % (snr))
            filepath = output_DIR + "Confusion Matrix (SNR=%d)" % snr + 'SNR.png'
            plt.savefig(filepath, format='png', dpi=600)
            plt.close()
            cor = np.sum(np.diag(cm))
            ncor = np.sum(cm) - cor
            print()
            print("SNR:{} Accuracy:{} ".format(snr, cor / (cor + ncor)))
            acc[snr] = 1.0 * cor / (cor + ncor)
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        logger.info(f'overall accuracy is: {acc}')
        logger.info(f'macro F1-score is: {F1_score}')
        logger.info(f'kappa coefficient is: {kappa}')

        if cfg.Draw_Acc_Curve is True:
            Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)
        if cfg.Draw_Confmat is True:
            Draw_Confmat(Confmat_Set, snrs, cfg)


def tsen_graphs_Run_Eval(model,
                         sig_test,
                         lab_test,
                         SNRs,
                         test_idx,
                         output_DIR,
                         cfg,
                         logger):
    global cm
    model.eval()
    with torch.no_grad():
        snrs = list(np.unique(SNRs))
        mods = list(cfg.classes.keys())

        Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
        Accuracy_list = np.zeros(len(snrs), dtype=float)

        # all 混淆矩阵
        Sample = torch.chunk(sig_test, cfg.test_batch_size, dim=0)
        Label = torch.chunk(lab_test, cfg.test_batch_size, dim=0)
        pred_all = []
        label_all = []
        for (sample, label) in zip(Sample, Label):
            sample = sample.to(cfg.device)
            logit, tsne, graphs = model(sample)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_all.append(pre_lab)
            label_all.append(label)
        pred_all = np.concatenate(pred_all)
        label_all = np.concatenate(label_all)
        plt.figure(figsize=(10, 10))
        cm = confusion_matrix(label_all, pred_all, normalize='pred')
        plot_confusion_matrix(cm, labels=mods)

        filepath = output_DIR + "confusion matrix" + '.png'
        plt.savefig(filepath, format='png', dpi=600)
        plt.close()
        # Classification Report
        report = classification_report(label_all, pred_all, target_names=mods)
        print(report)

        pre_lab_all = []
        label_all = []
        acc = {}
        for snr_i, snr in enumerate(snrs):
            test_SNRs = map(lambda x: SNRs[x], test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)
            pred_i = []
            label_i = []
            feature_map_list = []
            graphs_list = []
            labels = []
            for (sample, label) in zip(Sample, Label):
                sample = sample.to(cfg.device)
                logit, tsne, graphs = model(sample)

                pre_lab = torch.argmax(logit, 1).cpu()
                pred_i.append(pre_lab)
                label_i.append(label)
                for i in label.tolist():
                    tag = classestsen[i]
                    labels.append(tag)
                feature_map_list.append(tsne.cpu().detach().numpy())
                graphs_list.append(graphs.cpu().detach().numpy())
            feature_map_list = np.concatenate(feature_map_list)
            graphs_list = np.concatenate(graphs_list)
            ##保存graps
            np.save('graphs_list.npy', graphs_list)
            df = pd.DataFrame(labels, columns=["labels"])
            df.to_excel('labels.xlsx', index=True)


            t_sne(np.array(feature_map_list), np.array(labels), snr, output_DIR)
            pred_i = np.concatenate(pred_i)
            label_i = np.concatenate(label_i)

            pre_lab_all.append(pred_i)
            label_all.append(label_i)

            Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
            cm = confusion_matrix(label_i, pred_i)
            confnorm = np.zeros([len(mods), len(mods)])
            for i in range(0, len(mods)):
                confnorm[i, :] = cm[i, :] / np.sum(cm[i, :])
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(confnorm, labels=mods, title="Confusion Matrix (SNR=%d)" % (snr))
            filepath = output_DIR + "Confusion Matrix (SNR=%d)" % snr + 'SNR.png'
            plt.savefig(filepath, format='png', dpi=600)
            plt.close()
            cor = np.sum(np.diag(cm))
            ncor = np.sum(cm) - cor
            print()
            print("SNR:{} Accuracy:{} ".format(snr, cor / (cor + ncor)))
            acc[snr] = 1.0 * cor / (cor + ncor)
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        logger.info(f'overall accuracy is: {acc}')
        logger.info(f'macro F1-score is: {F1_score}')
        logger.info(f'kappa coefficient is: {kappa}')

        if cfg.Draw_Acc_Curve is True:
            Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)
        if cfg.Draw_Confmat is True:
            Draw_Confmat(Confmat_Set, snrs, cfg)


def Two_Run_Eval(model,
                 sig_test,
                 fb_test,
                 lab_test,
                 SNRs,
                 test_idx,
                 output_DIR,
                 cfg,
                 logger):
    global cm
    model.eval()
    with torch.no_grad():
        snrs = list(np.unique(SNRs))
        mods = list(cfg.classes.keys())

        Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
        Accuracy_list = np.zeros(len(snrs), dtype=float)

        # all 混淆矩阵
        Sample = torch.chunk(sig_test, cfg.test_batch_size, dim=0)
        Fb = torch.chunk(fb_test, cfg.test_batch_size, dim=0)
        Label = torch.chunk(lab_test, cfg.test_batch_size, dim=0)
        pred_all = []
        label_all = []
        for (sample, fb, label) in zip(Sample, Fb, Label):
            sample = sample.to(cfg.device)
            fb = fb.to(cfg.device)
            logit = model(sample, fb)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_all.append(pre_lab)
            label_all.append(label)
        pred_all = np.concatenate(pred_all)
        label_all = np.concatenate(label_all)
        plt.figure(figsize=(10, 10))
        cm = confusion_matrix(label_all, pred_all, normalize='pred')
        plot_confusion_matrix(cm, labels=mods)

        filepath = output_DIR + "confusion matrix" + '.png'
        plt.savefig(filepath, format='png', dpi=600)
        plt.close()
        # Classification Report
        report = classification_report(label_all, pred_all, target_names=mods)
        print(report)

        pre_lab_all = []
        label_all = []
        acc = {}
        for snr_i, snr in enumerate(snrs):
            test_SNRs = map(lambda x: SNRs[x], test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
            test_fb_i = fb_test[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
            Fb = torch.chunk(test_fb_i, cfg.test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)
            pred_i = []
            label_i = []
            for (sample, fb, label) in zip(Sample, Fb, Label):
                sample = sample.to(cfg.device)
                fb = fb.to(cfg.device)
                logit = model(sample, fb)
                pre_lab = torch.argmax(logit, 1).cpu()
                pred_i.append(pre_lab)
                label_i.append(label)
            pred_i = np.concatenate(pred_i)
            label_i = np.concatenate(label_i)

            pre_lab_all.append(pred_i)
            label_all.append(label_i)

            Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
            cm = confusion_matrix(label_i, pred_i)
            confnorm = np.zeros([len(mods), len(mods)])
            for i in range(0, len(mods)):
                confnorm[i, :] = cm[i, :] / np.sum(cm[i, :])
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(confnorm, labels=mods, title="Confusion Matrix (SNR=%d)" % (snr))
            filepath = output_DIR + "Confusion Matrix (SNR=%d)" % snr + 'SNR.png'
            plt.savefig(filepath, format='png', dpi=600)
            plt.close()
            cor = np.sum(np.diag(cm))
            ncor = np.sum(cm) - cor
            print()
            print("SNR:{} Accuracy:{} ".format(snr, cor / (cor + ncor)))
            acc[snr] = 1.0 * cor / (cor + ncor)
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        logger.info(f'overall accuracy is: {acc}')
        logger.info(f'macro F1-score is: {F1_score}')
        logger.info(f'kappa coefficient is: {kappa}')

        if cfg.Draw_Acc_Curve is True:
            Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)
        if cfg.Draw_Confmat is True:
            Draw_Confmat(Confmat_Set, snrs, cfg)
