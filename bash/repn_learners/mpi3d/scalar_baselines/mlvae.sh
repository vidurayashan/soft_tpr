for seed in 1828 251 434 1023 23
do 
    python src/baselines/scalar_tokened/vae_based/train.py --vae_beta 1 --save_ae --compute_dis_metrics --dataset mpi3d --checkpoint_freq 1000  --model mlvae --n_iters 200000 --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/mlvae-FINAL/
done