import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from ..models.custom_activations import TReLU, TReLU_with_trainable_bias


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def number_of_trainable_parameters(model, logger=print, verbose=True):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger(f" Number of total trainable parameters: {params}")
    return params


def neural_network_pruner(model, amount=0.6):

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # prune.remove(module, "weight")
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # prune.remove(module, "weight")


def first_layer_pruner(model, amount=0.8):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # prune.remove(module, "weight")
            break


def neural_network_sparsity(model, logger=print):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            logger("Sparsity in {:}.weight: {:.2f}%".format(module, 100. *
                                                            float(torch.sum(module.weight == 0)) / float(module.weight.nelement())))
        elif isinstance(module, torch.nn.Linear):
            logger("Sparsity in {:}.weight: {:.2f}%".format(module, 100. *
                                                            float(torch.sum(module.weight == 0)) / float(module.weight.nelement())))


def compute_l1_over_l2_of_weights(model, logger=print, verbose=True):
    l1_l2 = None
    count = 0
    epsilon = 0.0000001
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            l1_l2 = torch.mean(torch.norm(module.weight.view(
                module.weight.shape[0], -1), p=1, dim=-1)/(torch.norm(module.weight.view(module.weight.shape[0], -1), p=2, dim=-1) + epsilon))
            if verbose:
                logger("L1/L2 of weight of {:}: {:.2f}".format(module, l1_l2))
        elif isinstance(module, torch.nn.Linear):
            l1_l2 = torch.mean(torch.norm(module.weight.view(
                module.weight.shape[0], -1), p=1, dim=-1)/(torch.norm(module.weight.view(module.weight.shape[0], -1), p=2, dim=-1) + epsilon))
            if verbose:
                logger("L1/L2 of weight of {:}: {:.2f}".format(module, l1_l2))

        if l1_l2 and count == 0:
            first_layer_l1_l2 = l1_l2.detach().cpu().numpy()
            count += 1
    return first_layer_l1_l2


class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output.detach().cpu().numpy()

    def close(self):
        self.hook.remove()


def intermediate_layer_outputs_given_data(model, data, progress_bar=False):

    device = model.parameters().__next__().device
    first_layer = 0
    for name, module in model.named_modules():
        # breakpoint()
        if (isinstance(module, TReLU) or isinstance(module, nn.ReLU) or isinstance(module, TReLU_with_trainable_bias)) and first_layer == 0:
            # if isinstance(module, nn.Conv2d) and first_layer == 0:
            hookF = Hook(module)
            first_layer += 1

    # images = []
    activation_list = []

    out = model(data.to(device))
    activation_list.append(hookF.output)
    activation_list = np.concatenate(tuple(activation_list))

    hookF.close()
    return activation_list


def intermediate_layer_outputs(args, data_params, model, data_loader, device, progress_bar=True):

    first_layer = 0
    for name, module in model.named_modules():
        # breakpoint()
        # if (isinstance(module, TReLU) or isinstance(module, nn.ReLU) or isinstance(module, TReLU_with_trainable_bias)) and first_layer == 0:
        if isinstance(module, nn.Conv2d) and first_layer == 0:
            hookF = Hook(module)
            first_layer += 1

    images = []
    activation_list = []
    if progress_bar:
        iter_data_loader = tqdm(
            iterable=data_loader,
            unit="batch",
            leave=False)
    else:
        iter_data_loader = data_loader
    for X, y in iter_data_loader:
        X, y = X.to(device), y.to(device)

        out = model(X)
        images.append(X.detach().cpu().numpy())
        activation_list.append(hookF.output)

    images = np.concatenate(tuple(images))
    activation_list = np.concatenate(tuple(activation_list))
    return images, activation_list,


def intermediate_layer_adversarial_outputs(args, data_params, model, data_loader, device):
    from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT
    from deepillusion.torchattacks.analysis import whitebox_test
    from deepillusion.torchattacks.analysis.plot import loss_landscape
    # if isinstance(module, torch.nn.Conv2d):
    # register hooks on each layer

    first_layer = 0
    for name, module in model.named_modules():
        # breakpoint()
        if (isinstance(module, TReLU) or isinstance(module, nn.ReLU) or isinstance(module, TReLU_with_trainable_bias)) and first_layer == 0:
            hookF = Hook(module)
            first_layer += 1

    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM,
                   PGD_EOT=PGD_EOT)

    attack_params = {
        "norm": args.norm,
        "eps": args.epsilon,
        "alpha": args.alpha,
        "step_size": args.step_size,
        "num_steps": args.num_iterations,
        "random_start": args.rand,
        "num_restarts": args.num_restarts,
        "EOT_size": 10
        }

    adversarial_args = dict(attack=attacks[args.attack],
                            attack_args=dict(net=model,
                                             data_params=data_params,
                                             attack_params=attack_params,
                                             verbose=False))

    # frontend_output = []
    # frontend_output_adv = []
    activation_list = []
    activation_list_adv = []
    images = []
    images_adv = []
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)

        adversarial_args["attack_args"]["net"] = model
        adversarial_args["attack_args"]["x"] = X
        adversarial_args["attack_args"]["y_true"] = y
        # perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
        perturbs = torch.zeros_like(X)

        images.append(X.detach().cpu().numpy())

        X += perturbs

        images_adv.append(X.detach().cpu().numpy())

        # frontend_output.append(activation['frontend'])
        activation_list.append(hookF.output)

        out = model(X)

        activation_list_adv.append(hookF.output)
        # frontend_output_adv.append(activation['frontend'])

    images = np.concatenate(tuple(images))
    images_adv = np.concatenate(tuple(images_adv))
    # frontend_output = np.concatenate(tuple(frontend_output))
    # frontend_output_adv = np.concatenate(tuple(frontend_output_adv))
    activation_list = np.concatenate(tuple(activation_list))
    activation_list_adv = np.concatenate(tuple(activation_list_adv))

    return images, images_adv, activation_list, activation_list_adv


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
