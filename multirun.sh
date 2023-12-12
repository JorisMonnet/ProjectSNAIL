#!/bin/bash


echo "Running SNAIL with learning rate 0.00025"
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot dataset.set_cls.n_query=1 n_query=1 lr=0.00025

echo "Running SNAIL with learning rate 0.0005"
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot dataset.set_cls.n_query=1 n_query=1 lr=0.0005

echo "Running SNAIL with learning rate 0.00005"
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot dataset.set_cls.n_query=1 n_query=1 lr=0.00005
