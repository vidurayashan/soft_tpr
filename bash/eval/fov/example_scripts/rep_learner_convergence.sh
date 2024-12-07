# Example script to analyse representation learner convergence by examining downstream performance obtained at different iterations of rep learner training
# Note that COMET and VCT use different run_downstream_fov.py files to perform the downstream FoV regression
# Please refer to the table in Section 3.3 of README.md

# To make the dimensionality between models constant, simply pass arguments `--use_embed_layer` and `--desired_output_dim x`, where
# x is the desired output dim

for iter in 100 1000 10000 100000 200000
do 
    for load_dir in /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/cars3d/0/ae/iter_${iter}_soft_tpr_ae_dataset-cars3d_mod-None_latent_dim-1536_n_roles-10_embed_dim-12_n_fillers-106_embed_dim-128_weakly_supervised-lvq_1-lc_0.5-lr_1-lwsr_0.00024387238195519567-lwsa_0.0-lwsd_0.022006606405171873-None.pt \
    /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/cars3d/1/ae/iter_${iter}_soft_tpr_ae_dataset-cars3d_mod-None_latent_dim-1536_n_roles-10_embed_dim-12_n_fillers-106_embed_dim-128_weakly_supervised-lvq_1-lc_0.5-lr_1-lwsr_0.00024387238195519567-lwsa_0.0-lwsd_0.022006606405171873-None.pt \
    /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/cars3d/2/ae/iter_${iter}_soft_tpr_ae_dataset-cars3d_mod-None_latent_dim-1536_n_roles-10_embed_dim-12_n_fillers-106_embed_dim-128_weakly_supervised-lvq_1-lc_0.5-lr_1-lwsr_0.00024387238195519567-lwsa_0.0-lwsd_0.022006606405171873-None.pt \
    /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/cars3d/3/ae/iter_${iter}_soft_tpr_ae_dataset-cars3d_mod-None_latent_dim-1536_n_roles-10_embed_dim-12_n_fillers-106_embed_dim-128_weakly_supervised-lvq_1-lc_0.5-lr_1-lwsr_0.00024387238195519567-lwsa_0.0-lwsd_0.022006606405171873-None.pt \
    /media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/cars3d/4/ae/iter_${iter}_soft_tpr_ae_dataset-cars3d_mod-None_latent_dim-1536_n_roles-10_embed_dim-12_n_fillers-106_embed_dim-128_weakly_supervised-lvq_1-lc_0.5-lr_1-lwsr_0.00024387238195519567-lwsa_0.0-lwsd_0.022006606405171873-None.pt
    do
        python src/eval/fov_regression/run_downstream_fov.py --load_dir ${load_dir} --mode regression --repn_fn_key z_soft_tpr --wandb_proj_name rep_learner_convergence-fov --n_sample_list full,
    done
done
