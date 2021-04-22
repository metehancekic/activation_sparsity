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

    #--------------------------------------------------#
    #------------ Adversarial Argumenrs ---------------#
    #--------------------------------------------------#

    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM,
                   )

    attack_params = {
        "norm": args.attack_norm,
        "eps": args.attack_epsilon,
        "alpha": args.attack_alpha,
        "step_size": args.attack_step_size,
        "num_steps": args.attack_num_steps,
        "random_start": (
            args.at_rand and args.adv_training_num_restarts > 1
            ),
        "num_restarts": args.attack_num_restarts,
        "EOT_size": args.attack_EOT_size,
        }

    if "CWlinf" in args.adv_training_attack:
        adv_training_attack = args.adv_training_attack.replace(
            "CWlinf", "PGD")
        loss_function = "carlini_wagner"
    else:
        adv_training_attack = args.adv_training_attack
        loss_function = "cross_entropy"

    adversarial_args = adversarial_args = dict(
        attack=PGD,
        attack_args=dict(
            net=model, data_params=data_params, attack_params=attack_params
            ),
        loss_function=loss_function,
        )

    # Testing
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
    logger.info(f'Clean Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')
    test_loss, test_acc = NN.eval_model(adversarial_args=adversarial_args, progress_bar=True)
    logger.info(f'Attacked Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()
