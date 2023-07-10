#!/bin/bash
#SBATCH --job-name pr_full # 作业名为 example
#SBATCH --output job_log/pr_full_%J.out    # 屏幕上的输出文件重定向到 [JOBID].out
#SBATCH --gres gpu:a100:1  # 使用 1 张 A100 显卡
#SBATCH --requeue
#SBATCH --time 5-0
NAME=pr_ss_full
PYTHONUNBUFFERED="True"
echo $NAME
DATASET_PATH=/Data/semantic/full/train
TRAIN_BATCH_SIZE=2
LR=0.1
MODEL=Res16UNet34C
RUN_NAME=finetune_${NAME}_${LR}_${MODEL}
python -u train.py \
    --log_dir log \
    --seed 42 \
    --train_dataset Pheno4dVoxelization2cmDataset \
    --val_dataset Pheno4dVoxelization2cmtestDataset \
    --pheno4d_test_path /Data/semantic/full/val \
    --checkpoint_dir checkpoints \
    --num_workers 4 \
    --validate_step 50 \
    --optim_step 1 \
    --val_batch_size 1  \
    --save_epoch 1 \
    --max_iter 2000 \
    --scheduler PolyLR \
    --do_train \
    --run_name $RUN_NAME \
    --weights /pretrain_ckpt/pretrain_pheno4d_ckpt.pth \
    --model $MODEL \
    --lr $LR \
    --train_batch_size $TRAIN_BATCH_SIZE  \
    --pheno4d_path $DATASET_PATH \
    --wandb True
    