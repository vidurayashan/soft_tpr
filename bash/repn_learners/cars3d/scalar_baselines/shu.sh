for seed in 1927 8 1 2 18 36
do
    python src/baselines/scalar_tokened/shu/train.py --compute_dis_metrics --save_dir /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/shu/cars3d-FINAL/ --save_ae --seed ${seed} --gen_width 2 --dis_width 1 --gen_lr 1.191858125068826e-06 --dis_lr 1.0853570315993486e-06 --dis_cond_bias True  --dis_share_dense True --dis_uncond_bias False --enc_lr_mul 1 --enc_width 1 --dataset cars3d
done