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

    logger, args = init_logger()
    config, device = init_configuration(args)
    train_loader, test_loader, data_params = init_dataset(args)

    frontend = init_frontend(args)
    classifier = init_classifier(args)
    model = Combined(frontend, classifier).to(device)
    logger.info(model)

    classifier_filepath = classifier_ckpt_namer(args)
    classifier.load_state_dict(torch.load(classifier_filepath))

    frontend_filepath = autoencoder_ckpt_namer(args)
    frontend.load_state_dict(torch.load(frontend_filepath))

    # Testing
    NN = NeuralNetwork(model, args.classifier_arch)
    test_loss, test_acc = NN.eval_model(test_loader)
    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()
