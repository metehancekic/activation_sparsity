"""
Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""

import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .trades import trades_loss
from . import cross_entropy

from .torch_utils import intermediate_layer_outputs_given_data, get_lr

__all__ = ["NeuralNetwork", "adversarial_epoch", "adversarial_test"]


class NeuralNetwork(object):
    """
    Description:
        Neural network wrapper for training, testing, attacking

    init:
        model,
        name,
        optimizer,
        scheduler (optional),

    methods:
        train_model
        save_model
        load_model
        eval_model

    """

    def __init__(self, model, train_loader=None, test_loader=None, logger=None, num_epochs=100, log_interval=10, optimizer=None, scheduler=None, regularizer=None, regularizer_decay=1, summary_writer=None):
        super(NeuralNetwork, self).__init__()

        self.model = model
        try:
            self.name = model.name()
        except:
            self.name = "Unknown_model"

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = 100

        self.logger = logger
        self.log_interval = log_interval

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer
        self.regularizer_decay = regularizer_decay

        self.summary_writer = summary_writer

    def train_model(self, adversarial_args=None, verbose=True):
        if verbose:
            logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

        epoch_args = dict(adversarial_args=adversarial_args)

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_loss, train_acc = self.single_epoch(**epoch_args)

            if self.scheduler:
                lr = self.scheduler.get_lr()[0]
            else:
                lr = get_lr(self.optimizer)

            if self.summary_writer:
                weight_l2 = torch.norm(self.model.block1.conv.weight, p=2, dim=(1, 2, 3))
                weight_l1 = torch.norm(self.model.block1.conv.weight, p=1, dim=(1, 2, 3))
                biases = torch.abs(self.model.block1.relu.bias.squeeze())
                activations = intermediate_layer_outputs_given_data(
                    self.model, test_loader.__iter__().__next__()[0])

                self.summary_writer.add_histogram('Activation_distribution',
                                                  activations[activations > 0].reshape(-1), epoch)
                self.summary_writer.add_histogram('Filters_L1_Norm_Distribution', weight_l1, epoch)
                self.summary_writer.add_histogram('Filters_L1L2_Norm_Distribution',
                                                  weight_l1/(weight_l2+0.0000001), epoch)
                self.summary_writer.add_histogram(
                    'Bias_distribution_normalized_by_L1_of_weights', biases/(weight_l1+0.0000001), epoch)
                self.summary_writer.add_scalar('Loss/train', train_loss, epoch)
                self.summary_writer.add_scalar('Acc/train', train_acc, epoch)

            end_time = time.time()
            if verbose:
                logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')

                if epoch % log_interval == 0 or epoch == num_epochs:
                    test_loss, test_acc = self.eval_model(test_loader)
                    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

        if self.summary_writer:
            self.summary_writer.close()

    def save_model(self, checkpoint_dir):
        torch.save(self.model.state_dict(), checkpoint_dir)

    def load_model(self, checkpoint_dir):
        self.model.load_state_dict(torch.load(checkpoint_dir))

    def save_custom_model(self, model, checkpoint_dir):
        torch.save(model.state_dict(), checkpoint_dir)

    def load_custom_model(self, model, checkpoint_dir):
        model.load_state_dict(torch.load(checkpoint_dir))

    def eval_model(self, progress_bar=False, adversarial_args=None, save_blackbox=False, blackbox_save_location="/home/metehan/pytorch-tutorials/cifar/data/black_box_resnet"):

        device = self.model.parameters().__next__().device
        self.model.eval()

        perturbed_data = []
        perturbed_labels = []
        test_loss = 0
        test_correct = 0
        if progress_bar:
            iter_test_loader = tqdm(
                iterable=self.test_loader,
                unit="batch",
                leave=False)
        else:
            iter_test_loader = self.test_loader

        for data, target in iter_test_loader:
            data, target = data.to(device), target.to(device)

            # Adversary
            if adversarial_args and adversarial_args["attack"]:
                adversarial_args["attack_args"]["net"] = self.model
                adversarial_args["attack_args"]["x"] = data
                adversarial_args["attack_args"]["y_true"] = target
                perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
                data += perturbs

            output = self.model(data)

            if save_blackbox:
                perturbed_data.append(data.detach().cpu().numpy())
                perturbed_labels.append(target.detach().cpu().numpy())

            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=False)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

        if save_blackbox:
            perturbed_data = np.concatenate(tuple(perturbed_data))
            perturbed_labels = np.concatenate(tuple(perturbed_labels))

        test_size = len(test_loader.dataset)
        if save_blackbox:
            np.savez(blackbox_save_location,
                     perturbed_data, perturbed_labels)

        return test_loss/test_size, test_correct/test_size

    def single_epoch(self, adversarial_args):
        self.model.train()
        device = self.model.parameters().__next__().device

        train_loss = 0
        train_correct = 0
        for data, target in self.train_loader:

            data, target = data.to(device), target.to(device)

            # Adversary
            if adversarial_args and adversarial_args["attack"]:
                adversarial_args["attack_args"]["net"] = self.model
                adversarial_args["attack_args"]["x"] = data
                adversarial_args["attack_args"]["y_true"] = target
                perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
                data += perturbs

            # model.l1_normalize_weights()
            self.optimizer.zero_grad()
            output = model(data)
            cross_ent = nn.CrossEntropyLoss()
            loss = cross_ent(output, target)

            if self.regularizer = "l1":
                l1_penalty = torch.nn.L1Loss(size_average=False)
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += l1_penalty(param, torch.zeros_like(param))
                loss += self.regularizer_decay * reg_loss

            if self.regularizer = "hoyer":
                l1_penalty = torch.nn.L1Loss(size_average=False)
                l1_activations = l1_penalty(model.l1, torch.zeros_like(model.l1))
                l2_activations = torch.sum(model.l1**2, dim=(0, 1, 2, 3))
                loss += self.regularizer_decay*l1_activations/l2_activations

                # l1_activations = l1_penalty(model.l2, torch.zeros_like(model.l2))
                # l2_activations = torch.sum(model.l2**2, dim=(0, 1, 2, 3))
                # loss += l1_activations/l2_activations

                # l1_activations = l1_penalty(model.l3, torch.zeros_like(model.l3))
                # l2_activations = torch.sum(model.l3**2, dim=(0, 1))
                # loss += l1_activations/l2_activations

            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            train_loss += loss.item() * data.size(0)
            pred_adv = output.argmax(dim=1, keepdim=False)
            train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

        train_size = len(self.train_loader.dataset)

        return train_loss/train_size, train_correct/train_size
