
"""
Hyper-parameters
"""
import argparse
import os
from dotenv import load_dotenv, find_dotenv


def get_arguments():
    """ Hyper-parameters """

    load_dotenv(find_dotenv(), verbose=True)
    directory = os.getenv("PROJECT_DIR")

    parser = argparse.ArgumentParser(description="PyTorch Image Classification")

    # Directory
    parser.add_argument(
        "--directory",
        type=str,
        default=directory,
        metavar="",
        help="Directory of experiments",
        )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "Tiny-ImageNet", "Imagenette", "Imagenet", "mnist"],
        help="Directory of experiments",
        )

    neural_net = parser.add_argument_group(
        "neural_net", "Neural Network arguments")

    # Neural Model
    neural_net.add_argument(
        "--classifier_arch",
        type=str,
        default="VGG",
        metavar="classifier_name",
        help="Which classifier to use",
        )

    # Neural Model
    neural_net.add_argument(
        "--lr_scheduler",
        type=str,
        default="cyc",
        choices=["cyc", "mult", "step"],
        metavar="optimizer name",
        help="Which optimizer to use",
        )

    # Optimizer
    neural_net.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "rms"],
        metavar="optimizer name",
        help="Which optimizer to use",
        )

    neural_net.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="Learning rate",
        )
    neural_net.add_argument(
        "--lr_min", type=float, default=0.0, metavar="LR", help="Learning rate min",
        )
    neural_net.add_argument(
        "--lr_max", type=float, default=0.05, metavar="LR", help="Learning rate max",
        )
    neural_net.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="Optimizer momentum",
        )
    neural_net.add_argument(
        "--weight_decay", type=float, default=0.0005, metavar="WD", help="Weight decay",
        )

    neural_net.add_argument(
        "--save_attack",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to save the attack after training (default: False)",
        )
    # Batch Sizes & #Epochs
    neural_net.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        metavar="N",
        help="Batch size for train",
        )
    neural_net.add_argument(
        "--test_batch_size",
        type=int,
        default=100,
        metavar="N",
        help="Batch size for test",
        )
    neural_net.add_argument(
        "--classifier_epochs",
        type=int,
        default=100,
        metavar="N",
        help="Number of epochs",
        )

    neural_net.add_argument(
        "--regularizer",
        type=str,
        default="None",
        metavar="regularizer name",
        help="Which regularizer to use",
        )

    neural_net.add_argument(
        "--regularizer_decay",
        type=float,
        default=0.001,
        metavar="regularizer hypermeter",
        help="Regularizer hypermeter",
        )

    neural_net.add_argument(
        "-sm",
        "--save_checkpoint",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="For Saving the current Model, default = False ",
        )

    neural_net.add_argument("-gam", "--gamma", type=float, default=1.0,
                            metavar="", help="noise level",
                            )
    neural_net.add_argument("-pr", "--pruning", type=float, default=0.0,
                            metavar="", help="pruning level",
                            )
    neural_net.add_argument("-tk", "--top_k", type=int, default=20,
                            metavar="", help="top k",
                            )

    # Adversarial training parameters
    adv_training = parser.add_argument_group(
        "adv_training", "Adversarial training arguments"
        )

    adv_training.add_argument(
        "-tra",
        "--adv_training_attack",
        type=str,
        default="Standard",
        metavar="fgsm/pgd",
        help="Attack method",
        )
    adv_training.add_argument(
        "--adv_training_norm",
        type=str,
        default="inf",
        metavar="inf/p",
        help="Attack norm",
        )
    adv_training.add_argument(
        "-tr_eps",
        "--adv_training_epsilon",
        type=float,
        default=(8.0 / 255.0),
        metavar="",
        help="attack budget",
        )
    adv_training.add_argument(
        "-tr_a",
        "--adv_training_alpha",
        type=float,
        default=(10.0 / 255.0),
        metavar="",
        help="random fgsm budget",
        )
    adv_training.add_argument(
        "-tr_ss",
        "--adv_training_step_size",
        type=float,
        default=(1.0 / 255.0),
        metavar="",
        help="Step size for PGD, adv training",
        )
    adv_training.add_argument(
        "-tr_ns",
        "--adv_training_num_steps",
        type=int,
        default=10,
        metavar="",
        help="Number of steps for PGD, adv training",
        )
    adv_training.add_argument(
        "-tr_rand",
        "--adv_training_rand",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="randomly initialize attack for training",
        )
    adv_training.add_argument(
        "-tr_nr",
        "--adv_training_num_restarts",
        type=int,
        default=1,
        metavar="",
        help="number of restarts for pgd for training",
        )
    adv_training.add_argument(
        "-tr_eot",
        "--adv_training_EOT_size",
        type=int,
        default=1,
        metavar="",
        help="number of parallel models for EOT PGD",
        )

    # Adversarial testing parameters
    adv_testing = parser.add_argument_group(
        "adv_testing", "Adversarial testing arguments"
        )

    adv_testing.add_argument(
        "-at_method",
        "--attack_method",
        type=str,
        default="PGD",
        help="Attack method for white/semiwhite box attacks",
        )
    adv_testing.add_argument(
        "-at_norm",
        "--attack_norm",
        type=str,
        default="inf",
        metavar="inf/p",
        help="Which attack norm to use",
        )
    adv_testing.add_argument(
        "-at_eps",
        "--attack_epsilon",
        type=float,
        default=(8.0 / 255.0),
        metavar="",
        help="attack budget",
        )
    adv_testing.add_argument(
        "-at_a",
        "--attack_alpha",
        type=float,
        default=(10.0 / 255.0),
        metavar="",
        help="RFGSM step size",
        )
    adv_testing.add_argument(
        "-at_ss",
        "--attack_step_size",
        type=float,
        default=(1.0 / 255.0),
        metavar="",
        help="Step size for PGD",
        )
    adv_testing.add_argument(
        "-at_ni",
        "--attack_num_steps",
        type=int,
        default=40,
        metavar="",
        help="Number of steps for PGD",
        )

    adv_testing.add_argument(
        "--at_rand",
        action="store_true",
        default=False,
        help="randomly initialize PGD attack",
        )
    adv_testing.add_argument(
        "-at_nr",
        "--attack_num_restarts",
        type=int,
        default=1,
        metavar="",
        help="number of restarts for PGD",
        )
    adv_testing.add_argument(
        "-at_eot",
        "--attack_EOT_size",
        type=int,
        default=1,
        metavar="",
        help="number of parallel models for eot PGD",
        )
    adv_testing.add_argument(
        "--attack_progress_bar",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="show progress bar during PGD attack",
        )

    # Others
    others = parser.add_argument_group("others", "Other arguments")

    others.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
        )
    others.add_argument(
        "--seed", type=int, default=2021, metavar="S", help="random seed (default: 1)"
        )
    others.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
        )

    # Defense
    defense = parser.add_argument_group("defense", "Defense arguments")

    args = parser.parse_args()

    if args.lr_scheduler == "cyclic" and args.optimizer != "sgd":
        print("Cyclic learning rate can only be used with SGD.")
        raise AssertionError

    return args
