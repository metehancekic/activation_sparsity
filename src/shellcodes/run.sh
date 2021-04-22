#!/bin/bash 

# model=VGG_modified
# frontend=Identity

# COMMAND="python -m pytorch-tutorials.cifar.play_ground  \
# --model=$model --epochs 100 --frontend=$frontend -at -tr"
# echo $COMMAND
# eval $COMMAND

# model=VGG_all_trelu
# frontend=Identity

# COMMAND="python -m src.cifar.main  \
# --model=$model --epochs 100 --frontend=$frontend -at -tr"
# echo $COMMAND
# eval $COMMAND

model=VGG11_topk
frontend=Identity

COMMAND="python -m src.cifar.main \
--model=$model --epochs 20 --frontend=$frontend -tr  -at --test_batch_size=100"
echo $COMMAND
eval $COMMAND

# model=VGG2
# frontend=LP_Layer

# COMMAND="python -m src.cifar.main_2phase \
# --model=$model --epochs 100 --frontend=$frontend -tr -pr -at --test_batch_size=100"
# echo $COMMAND
# eval $COMMAND


# COMMAND="python -m src.cifar.main  \
# --model=$model --epochs 100 --frontend=$frontend -an --where_to_prune=all"
# echo $COMMAND
# eval $COMMAND

# model=VGG_wide_first_layer
# frontend=Identity

# COMMAND="python -m src.cifar.main  \
# --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=first"
# echo $COMMAND
# eval $COMMAND

# # COMMAND="python -m src.cifar.main  \
# # --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=all"
# # echo $COMMAND
# # eval $COMMAND

# model=VGG_all_trelu
# frontend=Identity

# COMMAND="python -m src.cifar.main  \
# --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=first"
# echo $COMMAND
# eval $COMMAND

# # COMMAND="python -m src.cifar.main  \
# # --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=all"
# # echo $COMMAND
# # eval $COMMAND

# model=VGG_single_trelu
# frontend=Identity

# COMMAND="python -m src.cifar.main  \
# --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=first"
# echo $COMMAND
# eval $COMMAND

# # COMMAND="python -m src.cifar.main  \
# # --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=all"
# # echo $COMMAND
# # eval $COMMAND

# model=VGG
# frontend=Identity

# COMMAND="python -m src.cifar.main  \
# --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=first"
# echo $COMMAND
# eval $COMMAND

# # COMMAND="python -m src.cifar.main  \
# # --model=$model --epochs 100 --frontend=$frontend -pr --where_to_prune=all"
# # echo $COMMAND
# # eval $COMMAND

# # model=VGG_wide_first_layer
# # frontend=Identity

# # COMMAND="python -m src.cifar.main  \
# # --model=$model --epochs 100 --frontend=$frontend -tr"
# # echo $COMMAND
# # eval $COMMAND

# # model=VGG_modified2
# # frontend=Identity

# # COMMAND="python -m pytorch-tutorials.cifar.main  \
# # --model=$model --epochs 100 --frontend=$frontend -at -tr"
# # echo $COMMAND
# # eval $COMMAND
