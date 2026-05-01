#!/bin/bash

# Generate 100 eps test data for 21 tasks 
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

for task in "${tasks[@]}"; do
  python RLBench/tools/dataset_generator.py --tasks="$task" \
                                            --save_path="$PERACT_ROOT/data/extended" \
                                            --image_size=128,128 \
                                            --renderer=opengl \
                                            --episodes_per_task=100 \
                                            --processes=1 \
                                            --all_variations=True
done