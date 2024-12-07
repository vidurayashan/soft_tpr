# Example script to analyse representation learner convergence by examining downstream performance obtained at different iterations of rep learner training
# Note that COMET and VCT use different run_downstream_fov.py files to perform the downstream FoV regression
# Please refer to the table in Section 3.3 of README.md
# Note that 100000 samples is equivalent to the 'full' dataset

# To make the dimensionality between models constant, simply pass arguments `--use_embed_layer` and `--desired_output_dim x`, where
# x is the desired output dim

for load_dir in /media/bethia/F6D2E647D2E60C251/trained_cleaning/0/ae/iter_200000_soft_tpr_ae_dataset-shapes3d_latent_dim-512_n_roles-10_embed_dim-16_n_fillers-57_embed_dim-32_ws-lvq_1-lc_0.5-lr_1-lwsd_0.002275725972461378.pt \
    /media/bethia/F6D2E647D2E60C251/trained_cleaning/1/ae/iter_200000_soft_tpr_ae_dataset-shapes3d_latent_dim-512_n_roles-10_embed_dim-16_n_fillers-57_embed_dim-32_ws-lvq_1-lc_0.5-lr_1-lwsd_0.002275725972461378.pt \
    /media/bethia/F6D2E647D2E60C251/trained_cleaning/2/ae/iter_200000_soft_tpr_ae_dataset-shapes3d_latent_dim-512_n_roles-10_embed_dim-16_n_fillers-57_embed_dim-32_ws-lvq_1-lc_0.5-lr_1-lwsd_0.002275725972461378.pt \
    /media/bethia/F6D2E647D2E60C251/trained_cleaning/3/ae/iter_200000_soft_tpr_ae_dataset-shapes3d_latent_dim-512_n_roles-10_embed_dim-16_n_fillers-57_embed_dim-32_ws-lvq_1-lc_0.5-lr_1-lwsd_0.002275725972461378.pt \
    /media/bethia/F6D2E647D2E60C251/trained_cleaning/4/ae/iter_200000_soft_tpr_ae_dataset-shapes3d_latent_dim-512_n_roles-10_embed_dim-16_n_fillers-57_embed_dim-32_ws-lvq_1-lc_0.5-lr_1-lwsd_0.002275725972461378.pt \
    /media/bethia/F6D2E647D2E60C251/trained_cleaning/5/ae/iter_200000_soft_tpr_ae_dataset-shapes3d_latent_dim-512_n_roles-10_embed_dim-16_n_fillers-57_embed_dim-32_ws-lvq_1-lc_0.5-lr_1-lwsd_0.002275725972461378.pt
do
    python src/eval/avr/run_downstream_avr.py --load_dir ${load_dir} --repn_fn_key z_soft_tpr --wandb_proj_name test_cleaning_wildnet-repn_learner_conv --n_sample_list 100000,10000,1000,500,250,100,
done