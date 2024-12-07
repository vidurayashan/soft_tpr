for seed in 97 16 283 1 71
do 
    python src/baselines/scalar_tokened/vae_based/train.py --save_ae --vae_beta 1 --dataset cars3d --checkpoint_freq 1000 --eval_frequency=20 --model gvae --compute_dis_metrics --n_iters 200000 --seed ${seed} --save_dir /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/adagvae_k_known-FINAL/
done