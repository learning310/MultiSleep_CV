import torch
from abc import abstractmethod
from numpy import inf
import numpy as np
import os


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, fold_id):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.fold_id = fold_id

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.checkpoint_dir = config.save_dir

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir=os.path.join(config.save_dir, "curve"),
            comment='',
            filename_suffix='',
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        all_outs = []
        all_trgs = []

        for epoch in range(self.epochs):
            result, epoch_outs, epoch_trgs = self._train_epoch(epoch, self.epochs - 1)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            all_outs.extend(epoch_outs)
            all_trgs.extend(epoch_trgs)

            self.logger.info(
                'Epoch:{:0>2d} Train_Loss:{:.3f} Valid_Loss:{:.3f} Train_Acc:{:.3f} Valid_Acc:{:.3f}'.format(
                    epoch,
                    log['loss'],
                    log['val_loss'],
                    log['accuracy'],
                    log['val_accuracy'],
                )
            )

            self.record_graph(log)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn\'t improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

        outs_name = "outs_" + str(self.fold_id)
        trgs_name = "trgs_" + str(self.fold_id)
        np.save(self.config._save_dir / outs_name, all_outs)
        np.save(self.config._save_dir / trgs_name, all_trgs)

        if self.fold_id == self.config["data_loader"]["args"]["num_folds"] - 1:
            self._calc_metrics()

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _calc_metrics(self):
        from sklearn.metrics import classification_report
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import os
        from os import walk

        n_folds = self.config["data_loader"]["args"]["num_folds"]
        all_outs = []
        all_trgs = []

        outs_list = []
        trgs_list = []
        save_dir = os.path.abspath(os.path.join(self.checkpoint_dir, os.pardir))
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if "outs" in file:
                    outs_list.append(os.path.join(root, file))
                if "trgs" in file:
                    trgs_list.append(os.path.join(root, file))

        if len(outs_list) == self.config["data_loader"]["args"]["num_folds"]:
            for i in range(len(outs_list)):
                outs = np.load(outs_list[i])
                trgs = np.load(trgs_list[i])
                all_outs.extend(outs)
                all_trgs.extend(trgs)

        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        names = ['W', "N1", "N2", "N3", "REM"]
        r = classification_report(
            all_trgs, all_outs, target_names=names, digits=6, output_dict=True
        )
        del r['accuracy']
        df = pd.DataFrame(r)
        df.loc["accuracy"] = accuracy_score(all_trgs, all_outs)
        df.loc["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df = df * 100
        df.loc["support"] = df.loc["support"] / 100
        file_name = self.config["name"] + "_classification_report.xlsx"
        report_Save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_Save_path)

        cm = confusion_matrix(all_trgs, all_outs)
        cm_file_name = self.config["name"] + "_confusion_matrix.torch"
        cm_Save_path = os.path.join(save_dir, cm_file_name)
        torch.save(cm, cm_Save_path)

    def record_graph(self, log):
        self.writer.add_scalars(
            "Loss", {"Train": log['loss'], "Valid": log['val_loss']}, log['epoch']
        )
        self.writer.add_scalars(
            "Acc",
            {"Train": log['accuracy'], "Valid": log['val_accuracy']},
            log['epoch'],
        )
