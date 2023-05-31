#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -J loss-patch-selection # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o /work/dlclarge2/rapanti-metassl-dino-stn/experiments/loss-patch-selection/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge2/rapanti-metassl-dino-stn/experiments/loss-patch-selection/log/%A.%a.%N.out  # STDERR
#SBATCH --array 0-3%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate torch

EXP_D=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/loss-patch-selection

# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
    run_pretrain_eval.py \
      --arch vit_tiny \
      --img_size 32 \
      --patch_size 4 \
      --local_crops_number 8 \
      --out_dim 4096 \
      --dataset CIFAR10 \
      --data_path /work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10 \
      --output_dir $EXP_D \
      --epochs 300 \
      --warmup_epochs 30 \
      --batch_size_per_gpu 128 \
      --use_fp16 on \
      --saveckp_freq 100 \
      --summary_writer_freq 100

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
