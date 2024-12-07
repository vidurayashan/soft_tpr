for seed in 134 12 82 98123 951
do 
    python src/baselines/scalar_tokened/vae_based/train.py --dataset cars3d --eval_frequency=20 --model slowvae --save_ae --n_iters 200000 --seed ${seed} --save_dir /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/mpi3d/slowvae/ --slowvae_gamma 1
done
 