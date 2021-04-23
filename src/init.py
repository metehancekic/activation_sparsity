"""

"""
import os
from pprint import pformat
import numpy as np
import logging
import json

# Torch
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


from .utils.read_datasets import get_loaders
from .utils.namers import classifier_log_namer
from .parameters import get_arguments


from .models.custom_models import topk_LeNet, topk_VGG
from .models import ResNet, VGG, ResNetWide


def init_args():
    args = get_arguments()
    return args


def init_logger(args, model_name):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(classifier_log_namer(model_name, args)),
            logging.StreamHandler()
            ])
    logger.info(pformat(vars(args)))
    logger.info("\n")

    return logger


def init_tensorboard(args, model_name):
    writer = SummaryWriter(args.directory + f"{args.dataset}/tensorboards/" +
                           classifier_params_string(model_name, args))
    return writer


def init_configuration(args):
    with open(args.directory + "configs/" + args.dataset + ".json") as config_file:
        config = json.load(config_file, encoding='utf-8')
    #--------------------------------------------------#
    #--------------- Seeds and Device -----------------#
    #--------------------------------------------------#
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return config, device


def init_dataset(args):
    train_loader, test_loader = get_loaders(args)
    x_min = 0.0
    x_max = 1.0
    data_params = {"x_min": x_min, "x_max": x_max}
    return train_loader, test_loader, data_params


def init_classifier(args):
    if args.classifier_arch.startswith("topk"):
        classifier = globals()[args.classifier_arch](gamma=args.gamma, k=args.top_k)
    else:
        classifier = globals()[args.classifier_arch]()
    return classifier


def init_optimizer_scheduler(args, config, model, batches_per_epoch, printer=print, verbose=True):

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            )
    elif args.optimizer == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
            )

    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
            )
    else:
        raise NotImplementedError

    if args.lr_scheduler == "cyc":
        lr_steps = args.classifier_epochs * batches_per_epoch
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config["lr_min"],
            max_lr=config["lr_max"],
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
            )
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[35], gamma=0.1
            )

    elif args.lr_scheduler == "mult":

        def lr_fun(epoch):
            if epoch % 3 == 0:
                return 0.962
            else:
                return 1.0

        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fun)
    else:
        raise NotImplementedError

    if verbose == True:
        printer(optimizer)
        printer(scheduler)

    return optimizer, scheduler
