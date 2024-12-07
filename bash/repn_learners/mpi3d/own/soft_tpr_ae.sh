for seed in 6 128 2 104 14
do 
    python src/repn_learners/soft_tpr_ae/train.py --wandb_proj_name check-FINAL-after-debug --compute_dis_metrics --supervision_mode ws --transition_prior locatello --dataset=mpi3d --eval_frequency=50 --freeze_role_embeddings --role_embed_dim=12 --n_roles=10 --filler_embed_dim=32 --n_fillers=50 --lambda_ws_recon 0 --lambda_ws_r_embed_ce 0. --lambda_ws_r_embed_ce 1.1605005601779466 --recon_loss_fn mse --seed ${seed}
done