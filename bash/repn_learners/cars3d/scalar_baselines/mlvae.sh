for seed in 4890 172 356 1 8
do 
    python src/baselines/scalar_tokened/vae_based/train.py --compute_dis_metrics --dataset cars3d --checkpoint_freq 1000 --eval_frequency=20 --model mlvae --n_iters 200000 --seed ${seed} --save_ae --save_dir /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/mlvae-FINAL/ --vae_beta 1
done