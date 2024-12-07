# 1. Reconstruction Loss Constants
BCE = 'bce'
MSE = 'mse'
RECON_LOSS_FN_CHOICES = [BCE, MSE]
MSE_RECON_LOSS = 'mse_recon_loss'
BCE_RECON_LOSS = 'bce_recon_loss'

# 2. Scheduler Loss Constants
COSINE = 'cosine'
STEP = 'step'
SCHEDULER_CHOICES = [COSINE, STEP]

# 3. Classifier Loss Constants
# BCE = 'bce' 
SOFT_HINGE = 'soft_hinge'

# 4. TPR Autoencoder Loss Constants
# 4.1 Loss types
TOTAL_LOSS = 'total_loss'
CE_LOSS = 'ce_loss'
RECON_LOSS = 'recon_loss'

# 4.2 Coefficients
ORTH_PENALTY_ROLE = 'orth_penalty_role'
ORTH_PENALTY_FILLER = 'orth_penalty_filler'
VQ_PENALTY = 'vq_penalty'
RECON_PENALTY = 'recon_penalty'
COMMITMENT_PENALTY = 'commitment_penalty'
WS_RECON_LOSS_PENALTY = 'ws_recon_loss_penalty'
WS_DIS_PENALTY = 'ws_dis_penalty'
L1_PENALTY = 'l1_penalty'

# 4.3 Miscellaneous and logging
SEMI_ORTH = 'semi_orth'
LIN_INDEP = 'lin_indep'

REGULARISATION = [SEMI_ORTH, LIN_INDEP]

ROLE_RANK = 'role_rank'
FILLER_RANK = 'filler_rank'