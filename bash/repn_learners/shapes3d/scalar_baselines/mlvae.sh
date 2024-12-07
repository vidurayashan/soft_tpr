for seed in 28 0 123 971 6
do 
    python src/baselines/scalar_tokened/vae_based/train.py --vae_beta 4 --compute_dis_metrics --dataset shapes3d --checkpoint_freq 1000 --eval_frequency=20 --model mlvae --save_ae --n_iters 200000 --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/mlvae_TUNED/
done