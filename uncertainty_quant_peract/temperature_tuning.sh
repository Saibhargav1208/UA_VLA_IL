#!/bin/bash

# Temp Scaling for Tasks in Training Set
tasks=(open_drawer 
        put_money_in_safe 
        reach_and_drag
        close_jar 
        insert_onto_square_peg 
        light_bulb_in 
        meat_off_grill  
        place_cups 
        place_shape_in_shape_sorter
        push_buttons 
        put_groceries_in_cupboard 
        put_item_in_drawer
        stack_blocks 
        stack_cups 
        turn_tap
        place_wine_at_rack_location 
        slide_block_to_color_target 
        sweep_to_dustpan_of_size)
        
## Need to Change demo_path, temperature.temp_log_root accordingly
for t in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python peract_reliability/train.py \
        method=PERACT_BC \
        rlbench.tasks=[$t]\
        rlbench.task_name='multi' \
        rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
        rlbench.demos=25 \
        rlbench.demo_path=$PERACT_ROOT/data/val \
        replay.batch_size=1 \
        replay.path=/tmp/replay \
        replay.max_parallel_processes=4 \
        method.voxel_sizes=[100] \
        method.voxel_patch_size=5 \
        method.voxel_patch_stride=5 \
        method.num_latents=2048 \
        method.transform_augmentation.apply_se3=True \
        method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
        method.pos_encoding_with_lang=False \
        framework.training_iterations=600000 \
        framework.num_weights_to_keep=60 \
        framework.start_seed=0 \
        framework.log_freq=1000 \
        framework.save_freq=1000 \
        framework.logdir=$PERACT_ROOT/ckpts/ \
        framework.csv_logging=True \
        framework.tensorboard_logging=True \
        ddp.num_devices=1 \
        scaler.type='temperature' \
        temperature.temperature_training=True \
        temperature.temperature_use_hard_temp=False \
        temperature.temperature_training_iter=1000 \
        temperature.temp_log_root=$PATH_TO_TEMP
done