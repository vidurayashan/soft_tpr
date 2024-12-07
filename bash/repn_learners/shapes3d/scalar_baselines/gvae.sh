for seed in 348 12 87 98 1827
do 
    python src/baselines/scalar_tokened/vae_based/train.py --vae_beta 4 --compute_dis_metrics --dataset shapes3d --checkpoint_freq 1000 --eval_frequency=20 --model gvae --save_ae --n_iters 200000 --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/gvae_TUNED/
done