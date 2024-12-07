from src.repn_learners.soft_tpr_ae.model.decoders import VQ_VAE_Decoder, VQ_VAE_Decoder2, BaseDecoder, BaseDecoder2, BaseDecoder3
from src.repn_learners.soft_tpr_ae.model.encoders import AblationEncoder, GTEncoder, ModularEncoder, VQ_VAE_Encoder
from src.repn_learners.soft_tpr_ae.model.soft_tpr_ae import SoftTPRAutoencoder
from src.eval.fov_regression.models import Clf, ModularClf
from src.shared.constants import *
from src.repn_learners.baselines.scalar_tokened.vae_based.models import AdaGVAE, GVAEModel, MLVAEModel, SlowVAE, AdaGVAE_K_Known
from src.repn_learners.baselines.scalar_tokened.shu.model import Encoder as ShuEncoder

baseline_maps = {
    ADAGVAE: AdaGVAE, 
    SLOWVAE: SlowVAE,
    GVAE: GVAEModel, 
    MLVAE: MLVAEModel,
    ADAGVAE_K_KNOWN: AdaGVAE_K_Known,
    SHU: ShuEncoder
}


encoders_map = {
    ABLATION_ENCODER: AblationEncoder, 
    GT_ENCODER: GTEncoder,
    MODULAR_ENCODER: ModularEncoder,
    VQ_VAE_ENCODER: VQ_VAE_Encoder,
}

decoders_map = {
    BASE_DECODER1: BaseDecoder, 
    BASE_DECODER2: BaseDecoder2,
    BASE_DECODER3: BaseDecoder3,
    VQ_VAE_DECODER1: VQ_VAE_Decoder,
    VQ_VAE_DECODER2: VQ_VAE_Decoder2,
}

clfs_map = {
    BASE_CLF: Clf, 
    MODULAR_CLF: ModularClf
}

autoencoders_map = {
    SOFT_TPR_AE: SoftTPRAutoencoder
}