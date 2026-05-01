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
    CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model-folder $PATH_TO_MODEL \
    --eval-datafolder $PATH_TO_RLBench_Data \
    --tasks $t \
    --eval-episodes 100 \
    --log-name test/1 \
    --device 0 \
    --headless \
    --model-name model_14.pth \
    --scaler_type temperature \
    --calibrating False \
    --calib_log_path $PATH_TO_TEMP \
    --tau 11 \
    --ua_action_log_dir $PATH_TO_LOG_BASE \
    --ua_action_enabled False
done