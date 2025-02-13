import argparse
import os.path
from torchstat import stat
import numpy as np
import torch
from torchsummary import summary
from models.PGNet import *
from data_loader.data_loader import Create_Data_Loader, Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
from util.evaluation import Run_Eval, tsen_Run_Eval, tsen_graphs_Run_Eval
from util.training import Trainer, CTrainer
from util.utils import fix_seed, log_exp_settings, create_AMC_Net
from util.logger import create_logger
from util.visualize import Visualize_ACM, save_training_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')  # train ,eval or visualize
    parser.add_argument('--dataset', type=str, default='2016.10a')  # 2016.10a, 2016.10b, 2018.01a,2022.01a
    parser.add_argument('--data_path', type=str, default='D:\pycharm\data/')
    parser.add_argument('--output_DIR', type=str, default='E:\pycharm\pytorch_train_my_all\AMC-Net-main\output/')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--ckpt_path', type=str,
                        default='E:\pycharm\pytorch_train_my_all\AMC-Net-main\\training\\2016.10a_615\models\\')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--Draw_Confmat', type=bool, default=True)
    parser.add_argument('--Draw_Acc_Curve', type=bool, default=True)
    args = parser.parse_args()

    fix_seed(args.seed)

    cfg = Config(args.dataset, train=(args.mode == 'train'))
    cfg = merge_args2cfg(cfg, vars(args))
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    log_exp_settings(logger, cfg)
    # 定义模型
    model = PGNet_AMC(num_classes=cfg.num_classes).to(cfg.device)
    model1 = PGNet_AMC(num_classes=cfg.num_classes)
    model2 = PGNet_AMC(num_classes=cfg.num_classes)
    # stat(model1, (1, 128, 2))
    summary(model, (1, 128, 2), depth=4)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(cfg.dataset, logger, args.data_path)
    train_set, test_set, test_idx = Dataset_Split(
        Signals,
        Labels,
        snrs,
        mods,
        logger)
    Signals_test, Labels_test = test_set

    if args.mode == 'train':
        train_loader, val_loader = Create_Data_Loader(train_set, test_set, cfg, logger)
        trainer = Trainer(model,
                          train_loader,
                          val_loader,
                          cfg,
                          logger)
        trainer.loop()

        save_training_process(trainer.epochs_stats, cfg)

        save_model_name = cfg.dataset + '_' + 'AMC_Net' + '.pkl'
        model.load_state_dict(torch.load(os.path.join(cfg.model_dir, save_model_name)))
        tsen_Run_Eval(model,
                      Signals_test,
                      Labels_test,
                      SNRs,
                      test_idx,
                      args.output_DIR,
                      cfg,
                      logger)
        model.eval()
        dump_input = torch.ones(1, 1, 128, 2).to(cfg.device)
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                             profile_memory=False) as prof:
            outputs = model(dump_input)
        print(prof.table())
        prof.export_chrome_trace('./resnet_profile.json')

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, cfg.dataset + '_' + 'AMC_Net' + '.pkl')))
        tsen_graphs_Run_Eval(model,
                             Signals_test,
                             Labels_test,
                             SNRs,
                             test_idx,
                             args.output_DIR,
                             cfg,
                             logger)

    elif args.mode == 'visualize':
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, cfg.dataset + '_' + 'AMC_Net' + '.pkl')))
        for i in range(0, 8):
            index = np.random.randint(0, Signals_test.shape[0], 16)
            test_sample = Signals_test[index]
            Visualize_ACM(model, test_sample, cfg, index)
