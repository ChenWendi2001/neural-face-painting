python train.py \
    --name painter \
    --gpu_ids 0 \
    --model finetune \
    --dataset_mode face \
    --dataroot C:\\Users\\12779\\Desktop\\CelebAMask-HQ \
    --num_threads 8 \
    --preprocess crop \
    --crop_size 32 \
    --batch_size 64 \
    --display_freq 1 \
    --print_freq 1 \
    --lr 1e-4 \
    --init_type normal \
    --n_epochs 10 \
    --n_epochs_decay 5 \
    --max_dataset_size 1536 \
    --save_epoch_freq 1