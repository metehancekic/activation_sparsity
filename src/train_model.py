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

    optimizer, scheduler = init_optimizer_scheduler(
        args, config, model, len(train_loader), printer=logger.info, verbose=True)

    _ = number_of_trainable_parameters(model=model, logger=logger.info, verbose=True)

    #--------------------------------------------------#
    #------------ Adversarial Argumenrs ---------------#
    #--------------------------------------------------#

    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM,
                   )

    attack_params = {
        "norm": args.adv_training_norm,
        "eps": args.adv_training_epsilon,
        "alpha": args.adv_training_alpha,
        "step_size": args.adv_training_step_size,
        "num_steps": args.adv_training_num_steps,
        "random_start": (
            args.adv_training_rand and args.adv_training_num_restarts > 1
            ),
        "num_restarts": args.adv_training_num_restarts,
        "EOT_size": args.adv_training_EOT_size,
        }

    if "CWlinf" in args.adv_training_attack:
        adv_training_attack = args.adv_training_attack.replace(
            "CWlinf", "PGD")
        loss_function = "carlini_wagner"
    else:
        adv_training_attack = args.adv_training_attack
        loss_function = "cross_entropy"

    adversarial_args = dict(
        attack=attacks[adv_training_attack],
        attack_args=dict(
            net=model, data_params=data_params, attack_params=attack_params
            ),
        loss_function=loss_function,
        )

    # adversarial_args = dict(attack=attacks[args.tr_attack],
    #                         attack_args=dict(net=model,
    #                                          data_params=data_params,
    #                                          attack_params=attack_params,
    #                                          verbose=False))

    # Actual training
    NN = NeuralNetwork(model, args.classifier_arch, optimizer, scheduler)
    NN.train_model(train_loader, test_loader, logger, epoch_type="Standard",
                   num_epochs=args.classifier_epochs, log_interval=args.log_interval,
                   adversarial_args=adversarial_args)

    # Save checkpoint
    # if args.save_model:
    #     # NN.save_custom_model(frontend, checkpoint_dir=ckpt_frontend)
    #     # NN.save_custom_model(classifier, checkpoint_dir=ckpt_classifier)
    #     NN.save_custom_model(model, checkpoint_dir=ckpt_classifier)
    if args.save_checkpoint:
        if not os.path.exists(args.directory + "checkpoints/classifiers/"):
            os.makedirs(args.directory + "checkpoints/classifiers/")

        classifier_filepath = classifier_ckpt_namer(args)
        torch.save(classifier.state_dict(), classifier_filepath)

        logger.info(f"Saved to {classifier_filepath}")

        if not args.no_autoencoder:
            if not os.path.exists(args.directory + "checkpoints/autoencoders/"):
                os.makedirs(args.directory + "checkpoints/autoencoders/")

            autoencoder_filepath = autoencoder_ckpt_namer(args)
            if args.autoencoder_train_supervised:
                torch.save(frontend.state_dict(), autoencoder_filepath)

            logger.info(f"Saved to {autoencoder_filepath}")


if __name__ == "__main__":
    main()
