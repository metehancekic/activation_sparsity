"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT

# CIFAR10 TRAIN TEST CODES
from .init import *
from .utils import NeuralNetwork, number_of_trainable_parameters
from .utils.namers import autoencoder_ckpt_namer, classifier_ckpt_namer
# from .gabor_trial import plot_image


def main():

    args = init_args()

    config, device = init_configuration(args)
    train_loader, test_loader, data_params = init_dataset(args)

    model = init_classifier(args).to(device)
    logger = init_logger(args, model.name())
    writer = init_tensorboard(args, model.name())

    logger.info(model)

    classifier_filepath = classifier_ckpt_namer(model.name(), args)
    model.load_state_dict(torch.load(classifier_filepath))

    NN_args = dict(model=model,
                   train_loader=train_loader,
                   test_loader=test_loader,
                   logger=logger,
                   num_epochs=args.classifier_epochs,
                   log_interval=args.log_interval,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   regularizer=args.regularizer,
                   regularizer_decay=args.regularizer_decay,
                   summary_writer=writer)

    NN = NeuralNetwork(**NN_args)
    test_loss, test_acc = NN.eval_model()
    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()
