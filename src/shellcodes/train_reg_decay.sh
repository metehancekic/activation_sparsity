#!/bin/bash 

# export CUDA_VISIBLE_DEVICES="2"
# export PYTHONPATH=/home/rfml/rfml/sourcecode/lib/

declare -a arr=(10 1 0.1 0.01 0.001 0.0001)

for i in "${arr[@]}"
do
	COMMAND="python -m src.train_classifier --dataset mnist --classifier_arch= --regularizer hoyer --regularizer_decay=$i --save_checkpoint"
	echo $COMMAND
	eval $COMMAND
done
