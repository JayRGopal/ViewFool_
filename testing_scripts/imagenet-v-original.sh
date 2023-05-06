#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -n 16
#SBATCH --mem=100G
#SBATCH --account=carney-tserre-condo
#SBATCH --partition gpu
#SBATCH --gres=gpu:4
#SBATCH -J imagenet-v-original
#SBATCH -o log-imagenet-v-original-%j.out

cd ~/Neurips2023/ViewFool_
module load anaconda/latest
module load gcc/10.2
module load python/3.9.0
source activate base


# Define an array of model paths to loop through
# MODEL_PATHS=(
#     "/cifs/data/tserre_lrs/projects/prj_video_imagenet/models_to_test/outs_finetune_e2D_d2D_pretrain_vitbase_patch16_224_IN_jump4_checkpoint-99.pth"
#     "/cifs/data/tserre_lrs/projects/prj_video_imagenet/models_to_test/outs_finetune_e2D_d3D_pretrain_vitbase_patch16_224_IN_jump16_checkpoint-99.pth"
#     "/cifs/data/tserre_lrs/projects/prj_video_imagenet/models_to_test/outs_finetune_e2D_d3D_pretrain_vitbase_patch16_224_IN_jump4_checkpoint-99.pth"
#     "/cifs/data/tserre_lrs/projects/prj_video_imagenet/models_to_test/outs_finetune_e2D_d3D_pretrain_vitbase_patch16_224_IN_jump8_checkpoint-99.pth"
# )

# Set the path to the directory containing the .pth files
PTH_DIR="/cifs/data/tserre_lrs/projects/prj_video_imagenet/models_to_test/"


MODEL_PATHS=($(find "$PTH_DIR" -name "*.pth"))

# Loop through each model path and execute the script for each one
for MODEL_NOW in "${MODEL_PATHS[@]}"
do
    RESULTS_FILE="Jay_output/results_$(basename $MODEL_NOW .pth).json"
    python3 ./NeRF/Imagenet_v_benchmark.py --model "$MODEL_NOW" --savepath "$RESULTS_FILE"
done
