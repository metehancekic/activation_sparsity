"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np
import matplotlib.pyplot as plt

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT

# CIFAR10 TRAIN TEST CODES
from .init import *
from .utils import NeuralNetwork, number_of_trainable_parameters
from .utils.namers import autoencoder_ckpt_namer, classifier_ckpt_namer
from .utils.torch_utils import intermediate_layer_outputs
# from .gabor_trial import plot_image


def main():

    logger, args = init_logger()
    config, device = init_configuration(args)
    train_loader, test_loader, data_params = init_dataset(args)

    frontend = init_frontend(args)
    model = init_classifier(args).to(device)
    logger.info(model)

    model_name = model.model_name()
    classifier_filepath = classifier_ckpt_namer(model_name, args)
    model.load_state_dict(torch.load(classifier_filepath))

    #--------------------------------------------------#
    #------------ Adversarial Argumenrs ---------------#
    #--------------------------------------------------#

    # Testing
    NN = NeuralNetwork(model, args.classifier_arch)
    test_loss, test_acc = NN.eval_model(
        test_loader, adversarial_args=None, progress_bar=True)
    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    images, activations = intermediate_layer_outputs(args, data_params, model, test_loader, device)

    plt.figure()

    plt.hist(activations.reshape(-1), 50, density=True, facecolor='g', alpha=0.75)
    plt.savefig("act_hist.pdf")

    breakpoint()


if __name__ == "__main__":
    main()
