# 1. Model Types
AUTOENCODER = 'autoencoder'
CLF = 'clf'
REGRESSOR = 'regressor'

model_to_save_prefix = {
    AUTOENCODER: 'ae', 
    CLF: CLF,
    REGRESSOR: 'reg'
}

# 2. Autoencoder Types 
SOFT_TPR_AE = 'tpr_ae'
VQ_VAE = 'vq_vae'
MODEL_CHOICES = [SOFT_TPR_AE, VQ_VAE]

# 2.1 Encoder Types 
ABLATION_ENCODER = 'ablation_encoder'
MODULAR_ENCODER = 'modular_encoder'
VQ_VAE_ENCODER = 'vq_vae_encoder'
TPR_ENCODER = 'tpr_encoder'
GT_ENCODER = 'gt_encoder'
ENCODER_CHOICES = [ABLATION_ENCODER, TPR_ENCODER, GT_ENCODER, MODULAR_ENCODER,
                   VQ_VAE_ENCODER]

# 2.2 Decoder Types 
BASE_DECODER1 = 'base_decoder1'
BASE_DECODER2 = 'base_decoder2'
BASE_DECODER3 = 'base_decoder3'
VQ_VAE_DECODER1 = 'vq_vae_decoder1'
VQ_VAE_DECODER2 = 'vq_vae_decoder2'
DECODER_CHOICES = [BASE_DECODER1, BASE_DECODER2, BASE_DECODER3, VQ_VAE_DECODER1, 
                   VQ_VAE_DECODER2]

# 3. Classifier Types
BASE_CLF = 'base_clf'
MODULAR_CLF = 'modular_clf'
FILLER_BINDER = 'filler_binder'
CLF_CHOICES = [BASE_CLF, MODULAR_CLF, FILLER_BINDER]

# 4. Regressor Types
READOUT_MLP = 'readout_mlp'

# 5. Baseline Models
PCL_MODEL = 'pcl'
SLOWVAE = 'slowvae'
VCT = 'vct'
SHU = 'shu'
COMET = 'comet'
ADAGVAE = 'adagvae'
ADAGVAE_K_KNOWN = 'adagvae_k_known'
GVAE = 'gvae'
MLVAE = 'mlvae'
WS_SCALAR_BASELINES = [ADAGVAE, ADAGVAE_K_KNOWN, GVAE, SLOWVAE, MLVAE, SHU]
US_VECTOR_BASELINES = [VCT, COMET]
BASELINES = WS_SCALAR_BASELINES + US_VECTOR_BASELINES

# 6. TPR Specific 
QUANTISED_FILLERS = 'quantised_fillers'
QUANTISED_FILLERS_SG = 'quantised_fillers_sg'
QUANTISED_FILLERS_CONCATENATED = 'quantised_fillers_concatenated'
SOFT_FILLERS = 'soft_fillers'
SOFT_FILLERS_CONCATENATED = 'soft_fillers_concatenated'
FILLER_IDXS = 'filler_idxs'
Z_TPR = 'z_tpr'
TPR_BINDINGS = 'tpr_bindings'
TPR_BINDINGS_FLATTENED = 'tpr_bindings_flattened'
Z_SOFT_TPR = 'z_soft_tpr'
CONCATENATED = 'concatenated'
FLATTENED = 'flattened'