for seed in 958 123 67 1 01923
do 
    python src/repn_learners/soft_tpr_ae/train.py --dataset cars3d  --wandb_proj_name check-FINAL-after-debug --compute_dis_metrics --n_iters=200000 --supervision_mode ws --transition_prior locatello --dataset=cars3d --freeze_role_embeddings --filler_embed_dim=128 --n_fillers=106 --compute_dis_metrics --role_embed_dim=12 --n_roles=10 --lambda_ws_recon 0.00024387238195519567 --lambda_ws_r_embed_ce 0.022006606405171873 --recon_loss_fn mse --seed ${seed}
done