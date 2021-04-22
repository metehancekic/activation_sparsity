"""
Example Run
python -m src.cifar.main  --model VGG11 -tr -sm
"""

import numpy as np

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT

# CIFAR10 TRAIN TEST CODES
from .init import *
from .utils import NeuralNetwork, number_of_trainable_parameters, first_layer_pruner
from .utils.namers import autoencoder_ckpt_namer, classifier_ckpt_namer
# from .gabor_trial import plot_image


def main():

    logger, args = init_logger()
    config, device = init_configuration(args)
    train_loader, test_loader, data_params = init_dataset(args)
    writer = init_tensorboard(args)

    model = init_classifier(args).to(device)
    if args.pruning > 0:
        first_layer_pruner(model, amount=args.pruning)
        print(f"Started with pruned network, amount:{args.pruning}")

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

    # Actual training
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
    NN.train_model(adversarial_args=adversarial_args)

    # Save checkpoint
    if args.save_checkpoint:
        if not os.path.exists(args.directory + "checkpoints/classifiers/"):
            os.makedirs(args.directory + "checkpoints/classifiers/")

        model_name = NN.name
        classifier_filepath = classifier_ckpt_namer(model_name, args)
        torch.save(model.state_dict(), classifier_filepath)

        logger.info(f"Saved to {classifier_filepath}")


if __name__ == "__main__":
    main()
