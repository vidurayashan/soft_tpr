for seed in 14 34 123 5 238
do 
    python src/baselines/scalar_tokened/vae_based/train.py --vae_beta 1 --dataset shapes3d --checkpoint_freq 1000 --save_ae --eval_frequency=20 --model adagvae_k_known --compute_dis_metrics --n_iters 200000 --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/adagvae_k_known-FINAL/
done