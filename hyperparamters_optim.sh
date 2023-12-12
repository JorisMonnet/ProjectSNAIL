#!/bin/bash

echo "Hyperparameter tuning using default architecture"

echo "Running SNAIL with learning rate 0.00025"
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot n_query=1 lr=0.00025

echo "Running SNAIL with learning rate 0.0005"
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot n_query=1 lr=0.0005

echo "Running SNAIL with learning rate 0.00005"
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot n_query=1 lr=0.00005
