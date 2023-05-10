# Instructions to as of 4/27/23

Make environment

    conda create --name viewfool -c conda-forge python=3.10 
    conda activate viewfool
    conda install pytorch 
    conda install -c pytorch torchvision 
    conda install -c conda-forge timm pytorch-lightning    
    conda install opencv imageio einops torch-optimizer kornia joblib pyzmq xlrd

Get repo

    git clone https://github.com/Heathcliff-saku/ViewFool_.git

run attack

    python NeRF/attack_randomsearch.py --dataset_name blender_for_attack --scene_name 'AP_random/apple_2' --img_wh 400 400 --N_importance 64 --ckpt_path './NeRF/ckpts/apple_2/epoch=29.ckpt' --num_sample 100 --optim_method random --search_num 6 --root_dir <insert path to nerf on oscar>