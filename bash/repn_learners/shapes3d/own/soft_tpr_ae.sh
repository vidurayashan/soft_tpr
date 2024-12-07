for seed in 999 124 2 192 5
do 
    python ./src/repn_learners/soft_tpr_ae/train.py --wandb_proj_name check-FINAL-after-debug --compute_dis_metrics --save_ae --save_dir /media/bethia/F6D2E647D2E60C251/trained_cleaning/ --supervision_mode ws --transition_prior locatello --dataset=shapes3d --freeze_role_embeddings --role_embed_dim=16 --n_roles=10 --filler_embed_dim=32 --lambda_ws_recon 0.0009090564479742856 --lambda_ws_r_embed_ce 0.002275725972461378 --recon_loss_fn mse --seed ${seed}
done 