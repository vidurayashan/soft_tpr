for seed in 134 951 498 11 72
do 
    python src/baselines/scalar_tokened/vae_based/train.py --dataset shapes3d --model slowvae --save_ae --n_iters 200000 --seed ${seed} --save_dir /media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/slowvae/ --slowvae_gamma 1
done