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

# # tasks not converged after 500
# tasks=( 
#     insert_onto_square_peg 
#     meat_off_grill 
#     place_cups 
#     place_shape_in_shape_sorter 
#     put_groceries_in_cupboard 
#     put_item_in_drawer 
#     reach_and_drag 
#     stack_cups 
#     push_buttons 
#     put_money_in_safe
# )


for t in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python train.py \
    --exp_cfg_path configs/all.yaml \
    --device 0 \
    --scaler_type temperature \
    --calibrating True \
    --calib_log_path /home/bobwu/shared/rvt/temp_train_v2 \
    --task $t \
    --calib_iters 1000 \
    # --refresh_replay
done