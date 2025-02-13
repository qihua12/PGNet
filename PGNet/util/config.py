import os.path

import yaml


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index


def merge_args2cfg(cfg, args_dict):
    for k, v in args_dict.items():
        setattr(cfg, k, v)
    return cfg


class Config:
    def __init__(self, dataset, train=True):
        self.dataset = dataset
        yaml_name = './config/%s.yml' % dataset
        if not os.path.exists(yaml_name):
            raise NotImplementedError(f"can not find cfg file: {yaml_name}")
        cfg = yaml.safe_load(open(yaml_name, 'r'))

        self.base_dir = 'training' if train else 'inference'
        os.makedirs(self.base_dir, exist_ok=True)

        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.patience = cfg['patience']
        self.milestone_step = cfg['milestone_step']
        self.gamma = cfg['gamma']
        self.lr = cfg['lr']

        self.num_classes = cfg['num_classes']
        self.sig_len = cfg["sig_len"]
        self.extend_channel = cfg["extend_channel"]
        self.latent_dim = cfg['latent_dim']
        self.num_heads = cfg["num_heads"]
        self.conv_chan_list = cfg["conv_chan_list"]

        self.monitor = cfg['monitor']
        self.test_batch_size = cfg['test_batch_size']

        if self.dataset == '2016.10a':
            self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                            'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
        elif dataset == '2016.10b':
            self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                            'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
        elif dataset == '2022.01a':
            self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                            'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
        elif dataset == '2016.10c':
            self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
                            'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
        elif dataset == '2018.01a':
            self.classes = {b'00K': 0, b'4ASK': 1, b'8ASK': 2, b'BPSK': 3, b'QPSK': 4,
                            b'8PSK': 5, b'16PSK': 6, b'32PSK': 7, b'16APSK': 8, b'32APSK': 9,
                            b'64APSK': 10, b'128APSK': 11, b'16QAM': 12, b'32QAM': 13, b'64QAM': 14,
                            b'128QAM': 15, b'256QAM': 16, b'AM-SSB-WC': 17, b'AM-SSB-SC': 18,
                            b'AM-DSB-WC': 19, b'AM-DSB-SC': 20, b'FM': 21, b'GMSK': 22, b'OQPSK': 23}
        else:
            raise NotImplementedError(f'Not Implement dataset:{self.dataset}')

        index = get_log_dir_index(self.base_dir)
        self.cfg_dir = '%s/%s' % (self.base_dir, self.dataset + index)
        self.model_dir = '%s/models' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.result_dir = '%s/result' % self.cfg_dir
        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
