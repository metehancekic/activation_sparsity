
#!/bin/bash 

export CUDA_VISIBLE_DEVICES="3"

model=VGG11_topk
gamma=1
top_k=640

COMMAND="python -m src.train_classifier  \
--classifier_arch=$model --classifier_epochs 100 -sm true --gamma=$gamma --top_k=$top_k"
echo $COMMAND
eval $COMMAND

COMMAND="python -m src.attack_classifier  \
--classifier_arch=$model --classifier_epochs 100 --attack_method PGD --attack_EOT_size 1 --attack_num_steps 40 --attack_num_restarts 1 --gamma=$gamma --top_k=$top_k"
echo $COMMAND
eval $COMMAND

# top_k=5

# COMMAND="python -m src.train_classifier  \
# --classifier_arch=$model --classifier_epochs 100 -sm true --gamma=$gamma --top_k=$top_k"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python -m src.attack_classifier  \
# --classifier_arch=$model --classifier_epochs 100 --attack_method PGD --attack_EOT_size 1 --attack_num_steps 40 --attack_num_restarts 1 --gamma=$gamma --top_k=$top_k"
# echo $COMMAND
# eval $COMMAND

# top_k=10

# COMMAND="python -m src.train_classifier  \
# --classifier_arch=$model --classifier_epochs 100 -sm true --gamma=$gamma --top_k=$top_k"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python -m src.attack_classifier  \
# --classifier_arch=$model --classifier_epochs 100 --attack_method PGD --attack_EOT_size 1 --attack_num_steps 40 --attack_num_restarts 1 --gamma=$gamma --top_k=$top_k"
# echo $COMMAND
# eval $COMMAND


# gamma=0

# COMMAND="python -m src.train_classifier  \
# --classifier_arch=$model --classifier_epochs 100 -sm true --gamma=$gamma -tra PGD"
# echo $COMMAND
# eval $COMMAND

# COMMAND="python -m src.attack_classifier  \
# --classifier_arch=$model --classifier_epochs 100 --attack_method PGD --attack_EOT_size 1 --attack_num_steps 40 --attack_num_restarts 1 --gamma=$gamma -tra PGD"
# echo $COMMAND
# eval $COMMAND
