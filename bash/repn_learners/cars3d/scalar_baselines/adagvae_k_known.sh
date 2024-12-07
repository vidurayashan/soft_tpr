for seed in 14 8671 1893 0 18
do 
    python train_ws_scalar_baselines.py --save_ae --vae_beta 1 --dataset cars3d --checkpoint_freq 1000 --eval_frequency=20 --model adagvae_k_known --compute_dis_metrics --n_iters 200000 --seed ${seed} --save_dir /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/adagvae_k_known-FINAL/
done