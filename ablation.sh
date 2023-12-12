#!/bin/bash

flatten_json() {
    echo "$1" | tr -d '\n\t ' 
}

architecture_default="
    [
        {
            module: attention,
            att_key_size: 64,
            att_value_size: 32
        },
        {
            module: tc,
            tc_filters: 128
        },
        {
            module: attention,
            att_key_size: 256,
            att_value_size: 128
        },
        {
            module: tc,
            tc_filters: 128
        },
        {
            module: attention,
            att_key_size: 512,
            att_value_size: 256
        }
    ]"


arch_attention="
    [
        {
            module: attention,
            att_key_size: 64,
            att_value_size: 32
        },
        {
            module: attention,
            att_key_size: 256,
            att_value_size: 128
        },
        {
            module: attention,
            att_key_size: 512,
            att_value_size: 256
        }
    ]"

arch_tc="
    [
        {
            module: tc,
            tc_filters: 128
        },
        {
            module: tc,
            tc_filters: 128
        },
    ]"

echo "Running attention only architecture"
arch_attention_flattened=$(flatten_json "$arch_attention")
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot n_query=1 lr=0.0001 method.architecture=$arch_attention_flattened

echo "Running tc only architecture"
arch_tc_flattened=$(flatten_json "$arch_tc")
python run.py exp.name=snail_lr_tuning method=snail dataset=swissprot n_query=1 lr=0.0001 method.architecture=$arch_tc_flattened