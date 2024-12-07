for seed in 159 2 3484 1 90
do 
    python src/baselines/scalar_tokened/vae_based/train.py --vae_beta 1 --save_ae --compute_dis_metrics --dataset mpi3d --checkpoint_freq 1000  --model gvae --n_iters 200000 --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/gvae-FINAL/
done