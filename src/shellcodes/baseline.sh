#!/bin/bash 



model=ResNetWide
frontend=Identity

COMMAND="python -m src.cifar.main_teacher \
--model=$model --epochs 100 --frontend=$frontend -tr -at --test_batch_size=100"
echo $COMMAND
eval $COMMAND
