#!/bin/bash

# Define the list of tasks
tasks=( 
    close_jar 
    insert_onto_square_peg 
    light_bulb_in 
    meat_off_grill 
    place_cups 
    place_shape_in_shape_sorter 
    put_groceries_in_cupboard 
    put_item_in_drawer 
    reach_and_drag 
    stack_blocks 
    stack_cups 
    turn_tap   
    slide_block_to_color_target 
    sweep_to_dustpan_of_size
    push_buttons 
    put_money_in_safe 
    place_wine_at_rack_location 
    open_drawer
)

for t in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python peract_reliability/eval.py \
        rlbench.tasks=[$t] \
        rlbench.task_name='multi' \
        rlbench.demo_path=$PERACT_ROOT/data/extended \
        framework.gpu=0 \
        framework.logdir=$PERACT_ROOT/ckpts/ \
        framework.start_seed=0 \
        framework.eval_envs=100 \
        framework.eval_from_eps_number=0 \
        framework.eval_episodes=100 \
        framework.csv_logging=True \
        framework.tensorboard_logging=True \
        framework.eval_type='last' \
        rlbench.headless=True \
        risk.tau=5 \
        risk.trans_conf_thresh=1.0e-6 \
        risk.rot_conf_thresh=0.014 \
        risk.search_size=20 \
        risk.search_step=2 \
        risk.log_dir=$PATH_TO_EVAL_RESULTS+"/base_${t}/" \
        risk.enabled=True \
        temperature.temperature_use_hard_temp=True \
        temperature.temperature_hard_temp=1 \
        scaler.type='temperature' \
        temperature.load_indiv_temp=False
done
