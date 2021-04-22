import torch
import os
import numpy as np


def adv_training_params_string(args):
    adv_training_params_string = ""
    if args.adv_training_attack:
        adv_training_params_string += f"_{args.adv_training_attack}"
        adv_training_params_string += (
            f"_eps_{np.int(np.round(args.adv_training_epsilon*255))}"
            )
        if "EOT" in args.adv_training_attack:
            adv_training_params_string += f"_Ne_{args.adv_training_EOT_size}"
        if "PGD" in args.adv_training_attack or "CW" in args.adv_training_attack:
            adv_training_params_string += f"_Ns_{args.adv_training_num_steps}"
            adv_training_params_string += (
                f"_ss_{np.int(np.round(args.adv_training_step_size*255))}"
                )
            adv_training_params_string += f"_Nr_{args.adv_training_num_restarts}"
        if "FGSM" in args.adv_training_attack:
            adv_training_params_string += (
                f"_a_{np.int(np.round(args.adv_training_alpha*255))}"
                )

    return adv_training_params_string


def classifier_params_string(model_name, args):
    classifier_params_string = model_name

    classifier_params_string += f"_{args.optimizer}"

    classifier_params_string += f"_{args.lr_scheduler}"

    if args.gamma > 0:
        classifier_params_string += f"_{args.gamma}"

    if args.pruning > 0:
        classifier_params_string += f"_{args.pruning}"

    if args.lr_scheduler == "cyc":
        classifier_params_string += f"_{args.lr_max:.4f}"
    else:
        classifier_params_string += f"_{args.lr:.4f}"

    if args.regularizer != "None":
        classifier_params_string += f"_reg_{args.regularizer}_{args.regularizer_decay}"

    classifier_params_string += adv_training_params_string(args)

    classifier_params_string += f"_ep_{args.classifier_epochs}"

    return classifier_params_string


def attack_params_string(args):
    attack_params_string = f"attack_"

    elif args.attack_box_type == "white":
        if not args.no_autoencoder:
            attack_params_string += f"_{args.attack_whitebox_type}"

        if not args.attack_norm == 'inf':
            attack_params_string += f"_L{args.attack_norm}"

        attack_params_string += f"_{args.attack_method}"

        attack_params_string += f"_eps_{np.int(np.round(args.attack_epsilon*255))}"

        if "EOT" in args.attack_method:
            attack_params_string += f"_Ne_{args.attack_EOT_size}"
        if "PGD" in args.attack_method or "CW" in args.attack_method:
            attack_params_string += f"_Ns_{args.attack_num_steps}"
            attack_params_string += f"_ss_{np.int(np.round(args.attack_step_size*255))}"
            attack_params_string += f"_Nr_{args.attack_num_restarts}"
        if "RFGSM" in args.attack_method:
            attack_params_string += f"_a_{np.int(np.round(args.attack_alpha*255))}"
        if args.attack_whitebox_type == "W-AGGA":
            attack_params_string += f"_sig_{args.ablation_blur_sigma:.2f}"
        if args.attack_whitebox_type == "W-NFGA" and args.attack_quantization_BPDA_steepness != 0.0:
            attack_params_string += f"_steep_{args.attack_quantization_BPDA_steepness:.1f}"
        if args.attack_whitebox_type == "top_T_top_U":
            attack_params_string += f"_U_{args.top_U}"

    return attack_params_string


def classifier_ckpt_namer(model_name, args):

    file_path = args.directory + f"checkpoints/classifiers/{args.dataset}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path += classifier_params_string(model_name, args)

    file_path += ".pt"

    return file_path


def classifier_log_namer(model_name, args):

    file_path = args.directory + f"logs/{args.dataset}/"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path += classifier_params_string(model_name, args)

    file_path += ".log"

    return file_path


def attack_file_namer(args):

    file_path = args.directory + f"data/attacked_dataset/{args.dataset}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path += attack_params_string(args)
    file_path += "_"
    if args.distill:
        file_path += "distill_"
    file_path += classifier_params_string(args)

    file_path += ".npy"

    return file_path


def attack_log_namer(args):

    file_path = args.directory + f"logs/{args.dataset}/"

    file_path += attack_params_string(args)
    file_path += "_"
    if args.distill:
        file_path += "distill_"
    file_path += classifier_params_string(args)

    file_path += ".log"

    return file_path
