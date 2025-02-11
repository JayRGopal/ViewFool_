#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -n 16
#SBATCH --mem=100G
#SBATCH --account=carney-tserre-condo
#SBATCH --partition gpu
#SBATCH --gres=gpu:4
#SBATCH -J viewfool-attack-apple
#SBATCH -o log-viewfool-attack-apple-%j.out

cd ~/Neurips2023/ViewFool_
module load anaconda/latest
module load gcc/10.2
module load python/3.9.0
source activate base


python3 NeRF/ViewFool.py --jay_result_base '/users/jgopal/Neurips2023/ViewFool_/resultsMay8/' --jay_model_path '/cifs/data/tserre_lrs/projects/prj_video_imagenet/models_to_test/outs_finetune_e2D_d3D_pretrain_vitbase_patch16_224_IN_jumpMAX_mask0.75_checkpoint-99.pth' --dataset_name blender_for_attack --scene_name  'outs_finetune_e2D_d3D_pretrain_vitbase_patch16_224_IN_jumpMAX_mask0.75_checkpoint-99/apple_2_real' --img_wh 400 400 --N_importance 64 --ckpt_path './NeRF/ckpts/apple_2/epoch=29.ckpt' --optim_method NES --search_num 6 --popsize 51 --iteration 100 --mu_lamba 0.01 --sigma_lamba 0.01 --num_sample 100 --label_name 'Granny Smith' --label 948 --root_dir '/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_video_imagenet/Gopal-ViewFool-Training/1.7apple_2/'
