for seed in 12 56 134 151 087
do 
    python src/baselines/scalar_tokened/vae_based/train.py --dataset mpi3d  --model slowvae --save_ae --n_iters 200000 --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/slowvae-FINAL/ --slowvae_gamma 1
done