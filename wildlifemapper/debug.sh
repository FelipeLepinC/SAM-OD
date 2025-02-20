#/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py --coco_path /mnt/mara/coco_1024_fixed --output_dir ./exp/box_model --batch_size 2 --num_workers 4
python train.py --yolo_path C:\Users\Asus\Documents\Feria\Repositorio\UTUAV-OD\Data\C_Split_80_10_10 --num_workers 0
