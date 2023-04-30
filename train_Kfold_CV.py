import torch
import argparse
import collections
import numpy as np
import models.loss as module_loss
import models.metric as module_metric
from data_loader.data_loaders import *
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *
from models.exp2 import Exp2
from models.exp3 import Exp3
from models.exp5 import Exp5

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = globals()[config["arch"]["type"]]()
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    data_loader, valid_data_loader = data_generator_np(
        folds_data[fold_id][0], folds_data[fold_id][1], batch_size
    )
    weights_for_each_class = [1.0, 1.0, 1.0, 1.0, 1.0]

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        fold_id=fold_id,
        valid_data_loader=valid_data_loader,
        class_weights=weights_for_each_class,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument(
        '-c',
        '--config',
        default="config.json",
        type=str,
        help='config file path (default: None)',
    )
    parser.add_argument(
        '-r',
        '--resume',
        type=str,
        default=None,
        help='path to latest checkpoint (default: None)',
    )
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default="0",
        help='indices of GPUs to enable (default: all)',
    )
    parser.add_argument('-f', '--folds', type=int, default=20, help='folds')
    parser.add_argument(
        '-da',
        '--np_data_dir',
        type=str,
        default='./data/',
        help='Directory containing numpy files',
    )

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    args = parser.parse_args()
    folds = int(args.folds)
    for fold_id in range(0, folds):
        options = []
        config = ConfigParser.from_args(parser, fold_id, options)
        folds_data = load_folds_data(
            args.np_data_dir, config["data_loader"]["args"]["num_folds"]
        )
        main(config, fold_id)
