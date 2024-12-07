from tqdm import tqdm 
import torch 
import logging 

from src.shared.constants import *
from src.shared.utils import get_filename

logging.basicConfig(level=logging.INFO)

def train_one_epoch(args, model, optimiser, train_loader, pb, 
                    update_frac: float, n_epoch: int, logger, max_iters: int, curr_iter: int) -> None: 
    model.train()
    for batch_idx, (imgs, factor_classes) in enumerate(train_loader): 
        imgs = imgs.cuda() 
        factor_classes = factor_classes.cuda()
        
        optimiser.zero_grad() 
        out = model.forward(imgs, factor_classes)
        loss = out['loss'][TOTAL_LOSS]
        loss.backward() 
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimiser.step() 
        
        loss_logs = out['loss']
        recon_loss_logs = dict(filter(lambda x: (BCE_RECON_LOSS in x[0] or MSE_RECON_LOSS in x[0]) and WEAKLY_SUPERVISED not in x[0], loss_logs.items()))
        other_logs = dict(filter(lambda x: (BCE_RECON_LOSS not in x[0] and MSE_RECON_LOSS not in x[0]) or WEAKLY_SUPERVISED not in x[0], loss_logs.items()))
        logger.log_scalars(logs={**recon_loss_logs, 
                                        'step': n_epoch*len(train_loader) + batch_idx}, 
                    prefix_to_append='model/loss/recon/train')
        logger.log_scalars(logs={**other_logs, 
                                        'step': n_epoch*len(train_loader) + batch_idx}, 
                    prefix_to_append='model/loss/specific/train')
        
        if batch_idx % 1000 == 0: 
            pb.set_postfix_str(
                f'Epoch: {n_epoch}\tTraining loss: {loss.item():.3f}\n'
            )
        pb.update(update_frac)
        curr_iter += 1 
        if args.save_ae: 
            model_file_name = get_filename(args)
            if (curr_iter  % args.checkpoint_freq == 0 and curr_iter != 0) or curr_iter == 100: 
                logger.save_model(args=args, model=model, model_file_name=model_file_name, iteration_id=curr_iter)

        if curr_iter > max_iters: 
            break 
    return out

def log_one_epoch(x: torch.Tensor, x_hat: torch.Tensor, n_epoch: int, logger, 
                  training: bool) -> None: 
    logger.log_reconstructions(x=x, x_hat=x_hat, n_epoch=n_epoch, training=training)
        
def eval_one_epoch(args, model, val_loader,
                   n_epoch: int, fixed_recon_batch: torch.Tensor, logger, 
                   mode: str='test', evaluate_avg_whole_dset: bool=False) -> None: 
    model.eval()
    mse_recon_losses = [] 
    bce_recon_losses =[] 
    logging.info(f"\n***EVALUATING MODEL at EPOCH {n_epoch}****")

    with torch.no_grad(): 
        for batch_idx, (imgs, factor_classes) in enumerate(tqdm(val_loader)):
            if batch_idx > 200 and evaluate_avg_whole_dset:  # evaluate over 200 batches to save time
                break  
            imgs = imgs.cuda() 
            factor_classes = factor_classes.cuda()
            out = model.forward(imgs, factor_classes)
            bce_recon_losses.append(out['loss'][f'{UNSUPERVISED}_{BCE_RECON_LOSS}'])
            mse_recon_losses.append(out['loss'][f'{UNSUPERVISED}_{MSE_RECON_LOSS}'])
            if not evaluate_avg_whole_dset: 
                loss_logs = out['loss']
                recon_loss_logs = dict(filter(lambda x: (BCE_RECON_LOSS in x[0] or MSE_RECON_LOSS in x[0]) and WEAKLY_SUPERVISED not in x[0], loss_logs.items()))
                other_logs = dict(filter(lambda x: (BCE_RECON_LOSS not in x[0] and MSE_RECON_LOSS not in x[0]) or WEAKLY_SUPERVISED in x[0], loss_logs.items()))

                logger.log_scalars(logs={**recon_loss_logs, 
                                                'step': batch_idx + (n_epoch//args.eval_frequency)*200}, 
                            prefix_to_append=f'model/loss/recon/{mode}')
                logger.log_scalars(logs={**other_logs, 
                                                'step': batch_idx + (n_epoch//args.eval_frequency)*200}, 
                            prefix_to_append=f'model/loss/specific/{mode}')
        avg_mse_loss = torch.tensor(mse_recon_losses).mean() 
        avg_bce_loss = torch.tensor(bce_recon_losses).mean() 
        
        if evaluate_avg_whole_dset: 
            logger.log_scalars(logs={f'avg_{UNSUPERVISED}_mse': avg_mse_loss, 
                                     f'avg_{UNSUPERVISED}_bce': avg_bce_loss}, 
                                     prefix_to_append=f'model/loss/final/{mode}')
            return avg_mse_loss, avg_bce_loss
            

    logging.info(f'Epoch: {n_epoch}\tVal loss (over 200 batches) (avg bce): {avg_bce_loss.item():.3f}')

    fixed_img, fixed_factor_cls = fixed_recon_batch
    out = model.forward(fixed_img.cuda(), fixed_factor_cls.cuda())
    return out

def train(args, train_loader, val_loader, full_loader, model, optimiser, scheduler, logger) -> None: 
    fixed_img, fixed_factor_cls = next(iter(val_loader))
    fixed_recon_batch = (fixed_img, fixed_factor_cls)
    pb = tqdm(total=args.n_iters, unit_scale=True, smoothing=0.1, ncols=70)
    update_frac = 1

    print(f'Update frac is {update_frac}')

    args.n_epochs = args.n_iters//len(train_loader) + 1
    curr_iter = 0 
    epoch_idx = 0
    
    while curr_iter < args.n_iters:
        print(f'Curr iter is {curr_iter}, n iters is {args.n_iters}')
        train_out = train_one_epoch(args=args, model=model, optimiser=optimiser, 
                                                  train_loader=train_loader, 
                        pb=pb, update_frac=update_frac, n_epoch=epoch_idx,
                        logger=logger, max_iters=args.n_iters, curr_iter=curr_iter)
        epoch_idx += 1 
        curr_iter += len(train_loader)
        if (epoch_idx - 1) % args.vis_frequency == 0: 
            log_one_epoch(x=train_out['state']['x'], x_hat=train_out['state']['x_hat'], n_epoch=epoch_idx, 
                        logger=logger, training=True)
            if args.supervision_mode == WEAKLY_SUPERVISED and f'{WEAKLY_SUPERVISED}_x_hat' in train_out['state'].keys(): 
                logger.log_swapped_reconstructions(x=train_out['state']['x'], 
                                            x_hat=train_out['state'][f'{WEAKLY_SUPERVISED}_x_hat'], n_epoch=epoch_idx, training=True)
        
        if (epoch_idx - 1) % args.eval_frequency == 0:
            val_out = eval_one_epoch(args, model=model, val_loader=val_loader, n_epoch=int(epoch_idx-1/args.eval_frequency), 
                                fixed_recon_batch=fixed_recon_batch, logger=logger)
            if args.vis_frequency > 0 and (epoch_idx - 1) % args.vis_frequency == 0: 
                log_one_epoch(x=val_out['state']['x'], x_hat=val_out['state']['x_hat'], n_epoch=epoch_idx,
                            logger=logger, training=False)
        if scheduler is not None: 
            scheduler.step() 

    test_avg_mse, test_avg_bce = eval_one_epoch(args=args, model=model, val_loader=val_loader, 
                                                n_epoch=epoch_idx, fixed_recon_batch=None, 
                                                logger=logger, mode='test', evaluate_avg_whole_dset=True)
    train_avg_mse, train_avg_bce = eval_one_epoch(args=args, model=model, val_loader=train_loader, n_epoch=epoch_idx, 
                                                  fixed_recon_batch=None, logger=logger, 
                                                  mode='train', evaluate_avg_whole_dset=True)
    # log differences
    diffs = list(map(
                lambda x: 
                    (x[1] - x[0])/x[0], 
                    zip([train_avg_mse, train_avg_bce], [test_avg_mse, test_avg_bce])
                )
            )
    diffs = {f'diff_mse': diffs[0], f'diff_bce': diffs[1]}
    logger.log_scalars(diffs, prefix_to_append='model/loss/final/')
    
    
    return {f'final_train_mse': train_avg_mse, 
            f'final_train_bce': train_avg_bce, 
            f'final_test_mse': test_avg_mse, 
            f'final_test_bce': test_avg_bce, 
            **diffs}
         