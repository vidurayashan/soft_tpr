for seed in 290 123 58 23 9
do 
    python src/baselines/scalar_tokened/vae_based/train.py --save_ae --vae_beta 1 --dataset mpi3d --checkpoint_freq 1000  --model adagvae_k_known --n_iters 200000 --compute_dis_metrics --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/adagvae_k_known-FINAL/
done