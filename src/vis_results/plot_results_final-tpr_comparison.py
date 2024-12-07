

# first parse out the results from the wandb log files because I was silly and didn't save these
# metrics to a file 
# metadata contains checkpoint dir
# {"args": ["--load_dir", "dir...."]}

import os
import copy
import json 
from typing import Dict, List
from src.shared.constants import *
import matplotlib.pyplot as plt 
import seaborn as sns 
import argparse 
import numpy as np
import itertools

np.set_printoptions(precision=3, suppress=True)

BASELINES = [VCT, COMET, ADAGVAE, GVAE, MLVAE, SHU, SLOWVAE]

def parse_results_from_wandb_logs(mode: str,
                                  model_type_to_load_dir_parent_map: Dict, 
                                  iters_to_plot: List[int], 
                                  models_to_plot: List[str],
                                  soft_tpr_variants_to_plot: List[str],
                                  wandb_log_path: str = '/home/bethia/Documents/Code/Disentanglement/TPR/NewExp/baselines_schott/integrate_other_baselines/wandb/') -> Dict: 
    paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(wandb_log_path) for f in fn if 'wandb-metadata.json' in f]
    results = {model_type: {iter_to_plot: [] for iter_to_plot in iters_to_plot} for model_type in models_to_plot}
    results = {**results, **{f'{model_type}_embed': {iter_to_plot: [] for iter_to_plot in iters_to_plot} for model_type in models_to_plot if model_type != SOFT_TPR_AE}}
    if SOFT_TPR_AE in models_to_plot and 'dis' not in mode:
        results = {**results, **{f'{SOFT_TPR_AE}_{repn_fn_key}': {iter_to_plot: [] for iter_to_plot in iters_to_plot} for repn_fn_key in [SOFT_FILLERS_CONCATENATED, QUANTISED_FILLERS_CONCATENATED,
                                               Z_SOFT_TPR, Z_TPR, FILLER_IDXS]}}
    
    for path in paths: 
        with open(path) as f: 
            args = json.load(f)["args"]
            try:
                idx = args.index("--load_dir")
            except: 
                continue
            load_dir = args[idx+1]
            for model_type, interested_dir in model_type_to_load_dir_parent_map.items(): 
                if model_type not in models_to_plot: 
                    continue
                if any(map(load_dir.__contains__, interested_dir)): 
                    if 'own_model' in load_dir.lower() and MPI3D_DATASET in load_dir.lower() and 'iter_1000_' in load_dir.lower():
                       if is_incorrect_run(load_dir): 
                           continue
                    if 'model_' in load_dir and (COMET in load_dir.lower() or VCT in load_dir.lower()):
                        saved_iter = int(load_dir.split('model_')[1].split('.')[0])
                    else:
                        saved_iter = int(load_dir.split('iter_')[1].split('_')[0])
                    if saved_iter not in iters_to_plot: 
                        continue
                    relevant_path = os.path.join(os.path.dirname(path), 'wandb-summary.json')
                    if not os.path.exists(relevant_path):
                        continue
                    if '--use_embed_layer' in args and 'dis' not in mode: 
                        results[f'{model_type}_embed'][saved_iter].append(get_result_from_summary_json(relevant_path))
                        break 
                    if '--repn_fn_key' in args and model_type == SOFT_TPR_AE and 'dis' not in mode: 
                        repn_fn_key = args[args.index('--repn_fn_key') + 1]
                        results[f'{model_type}_{repn_fn_key}'][saved_iter].append(get_result_from_summary_json(relevant_path))
                        break 
                    results[model_type][saved_iter].append(get_result_from_summary_json(relevant_path))

    if 'dis' not in mode:
        results.pop(SOFT_TPR_AE)
        results = dict(filter(lambda x: 'new_ae' not in x[0] or ('new_ae' in x[0] and x[0] in soft_tpr_variants_to_plot), 
                          results.items()))
    return results # {model_type: {saved_iter0: [results], saved_iter1: [results], ..., saved_itern: [results]}}

def is_incorrect_run(load_dir): 
    return not('latent_dim-384_n_roles-10_embed_dim-12_n_fillers-50_embed_dim-32_weakly_supervised-lvq_1-lc_0.5-lr_1-lwsr_0.0-lwsa_0.0-lwsd_1.1605005601779466-None' in load_dir)

def get_result_from_summary_json(summary_json_path) -> Dict: 
    with open(summary_json_path) as f: 
        results = json.load(f)
        return results


def get_results_to_plot(results: Dict, metric: str): 
    results_to_plot = results.copy()
    
    for model_name, iter_to_result_list in results.items(): 
        if 'diff' not in metric: 
            for iter, result_list in iter_to_result_list.items(): 
                filtered_result_list = []
                for result in result_list: 
                    try: 
                        result_of_interest = result[metric]
                        if result_of_interest is not None:
                            filtered_result_list.append(result_of_interest)
                    except:
                        pass 
                results_to_plot[model_name][iter] = filtered_result_list
    
    for model_name, iter_to_result_list in results_to_plot.items():
        print(f'***MODEL***{model_name}:\n')
        print(iter_to_result_list)
        print(f'****DONE****\n')
    results_to_plot_standard = dict(filter(lambda x: 'embed' not in x[0] and x[0], results_to_plot.items()))
    results_to_plot_embedded = dict(filter(lambda x: 'embed' in x[0], results_to_plot.items()))
    soft_tpr_results = dict(filter(lambda x: 'new_ae' in x[0], results_to_plot.items()))
    results_to_plot_embedded = {**results_to_plot_embedded, **soft_tpr_results}
    #print(f'Results to plot standard: \n{results_to_plot_standard}\n\n')
    #print(f'Results to plot embedded: \n{results_to_plot_embedded}\n\n')
    return results_to_plot_standard, results_to_plot_embedded

def add_vct_results(results_to_plot_standard: Dict, dataset: str, metric: str): 
    with open(f'src/vis_results/vct_dis_metric_results/vct_{dataset}_results.json') as f: 
        d = json.load(f)
    for iter_key, result_list in d.items():
        for result_dict in result_list: 
            iter = int(iter_key.split('_')[-1])
            if 'beta' in metric: 
                result = result_dict['beta_VAE_score']['eval_accuracy']
            elif 'fac' in metric: 
                result = result_dict['factor_VAE_score']['eval_accuracy']
            elif 'dci' in metric: 
                result = result_dict['dci_score']['disentanglement']
            elif 'mig' in metric: 
                result = result_dict['MIG_score']['discrete_mig']
            results_to_plot_standard[VCT][iter].append(result)

def line_plot_repn_learner_convergence_avr(results_by_dataset, title, models_to_plot, plot_save_path): 
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=25)
    metrics = ['avg_acc_n_samples_100000+easy3_test']
    sns.set_style('darkgrid', {'grid.linestyle': ':'})
    for metric in metrics:
        fig_e, axs_e = plt.subplots(1, 1, figsize=(15, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.06, 
                                    right=0.90,
                                    bottom=0.15, 
                                    top=0.95, 
                                    wspace=0.08,
                                    hspace=0.55)
        fig_s, axs_s = plt.subplots(1, 1, figsize=(15, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.06, 
                                    right=0.90,
                                    bottom=0.15, 
                                    top=0.95, 
                                    wspace=0.08,
                                    hspace=0.55)

        results_to_plot_standard, results_to_plot_embedded = get_results_to_plot(copy.deepcopy(results_by_dataset[SHAPES3D_DATASET]), metric)
        for prefix, axs, fig, to_plot in zip(['standard', 'embedded'], [axs_s, axs_e], 
                                             [fig_s, fig_e], 
                                             [results_to_plot_standard, results_to_plot_embedded]):
            to_plot = sorted(to_plot.items(), key=lambda x: SORT_ORDER[x[0]])
            to_plot = {k: v for (k, v) in to_plot}
            for j, (model_name, result_dict) in enumerate(to_plot.items()): 
                print(f'****** MODEL {model_name} *********')
                iters_to_plot = list(result_dict.keys())
                mus = [] 
                sigmas = []
                for _, results_list in result_dict.items(): 
                    mus.append(np.array(results_list).mean())
                    sigmas.append(np.array(results_list).std())
                mus = np.array(mus)
                sigmas = np.array(sigmas)
                print(f'For model {model_name}:') 
                formatted_table_line = ''
                for (mu, sigma) in zip(mus, sigmas): 
                    formatted_table_line += ' $\pm$ '.join([str(np.round(mu, 3)), str(np.round(sigma, 3))])
                    formatted_table_line += ' & '
                print(formatted_table_line)
                fig.gca().plot(iters_to_plot, mus, c=model_to_colour_map[model_name], linestyle='--', marker='o', label=j)
                fig.gca().fill_between(iters_to_plot, mus-sigmas, mus+sigmas, alpha=0.3, facecolor=model_to_colour_map[model_name])
                fig.gca().set_xscale('log')
                fig.gca().set_ylabel(metric_to_y_axis_lab[metric])
                fig.gca().set_xlabel('n epochs')
                plt.locator_params('x', len(iters_to_plot))
                fig.gca().title.set_text('Abstract visual reasoning dataset')
            fig.legend(*fig.gca().get_legend_handles_labels(), loc='upper right')
            frmted_models = '-'.join([pretty_models[model] for model in models_to_plot])
            fig_name = os.path.join(plot_save_path, f'{title_to_save_title[title]}_{prefix}_{frmted_models}_{metric_to_save_title[metric]}.pdf')
            print(f'Saving fig to...{fig_name}')
            fig.savefig(fname=fig_name, format='pdf')


import matplotlib 
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
def line_plot_repn_learner_convergence(downstream_task, results_by_dataset, mode: str, title: str, 
                                       models_to_plot: List[str], plot_save_path: str='/home/bethia/Documents/Code/Disentanglement/TPR/NewExp/baselines_schott/integrate_other_baselines/src/vis_results/plots/final') -> None: 
    if 'downstream' in mode: 
        if downstream_task != 'regression':
            line_plot_repn_learner_convergence_avr(results_by_dataset, title, 
                                                   models_to_plot, plot_save_path)
            return 
        metrics = ['reg_mlp/test_full_rsquared_excl_cat']
    else: 
        metrics = ['dmetric/train/_beta_val_acc', 
                   'dmetric/train/_fac_eval', 
                   'dmetric/train/_discrete_mig', 
                   'dmetric/train/_dci_disentanglement']
    
    sns.set_style('darkgrid', {'grid.linestyle': ':'})

    font = {'size': 15}
    matplotlib.rc('font', **font)

    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=25)
    
    for metric in metrics:
        for i, (dataset, results_dict) in enumerate(results_by_dataset.items()): 
            fig_e, axs_e = plt.subplots(1, figsize=(15, 8)) # one plot per dataset
            if dataset == CARS3D_DATASET:
                plt.subplots_adjust(left=0.075, 
                                    right=0.90,
                                    bottom=0.15, 
                                    top=0.95, 
                                    wspace=0.08,
                                    hspace=0.55)
            else: 
                plt.subplots_adjust(left=0.06, 
                                    right=0.90,
                                    bottom=0.15, 
                                    top=0.95, 
                                    wspace=0.08,
                                    hspace=0.55)
            fig_s, axs_s = plt.subplots(1, figsize=(15, 8)) # one plot per dataset
            plt.subplots_adjust(left=0.06, 
                                    right=0.90,
                                    bottom=0.15, 
                                    top=0.95, 
                                    wspace=0.08,
                                    hspace=0.55)
            results_to_plot_standard, results_to_plot_embedded = get_results_to_plot(copy.deepcopy(results_dict), metric)
            if 'dis' in mode and VCT in models_to_plot: 
                add_vct_results(results_to_plot_standard, dataset, metric)
            for prefix, axs, fig, to_plot in zip(['standard', 'embedded'], [axs_s, axs_e], 
                                                 [fig_s, fig_e], 
                                                 [results_to_plot_standard, results_to_plot_embedded]):
                to_plot = sorted(to_plot.items(), key=lambda x: SORT_ORDER[x[0]])
                to_plot = {k: v for (k, v) in to_plot}
                for j, (model_name, result_dict) in enumerate(to_plot.items()): 
                    print(f'****** MODEL {model_name} *********')
                    iters_to_plot = list(result_dict.keys())
                    mus = [] 
                    sigmas = []
                    for _, results_list in result_dict.items(): 
                        mus.append(np.array(results_list).mean())
                        sigmas.append(np.array(results_list).std())
                    mus = np.array(mus)
                    sigmas = np.array(sigmas)
                    formatted_table_line = ''
                    for (mu, sigma) in zip(mus, sigmas): 
                        formatted_table_line += ' $\pm$ '.join([str(np.round(mu, 3)), str(np.round(sigma, 3))])
                        formatted_table_line += ' & '
                    print(formatted_table_line)
                    axs.plot(iters_to_plot, mus, c=model_to_colour_map[model_name], linestyle='--', marker='o', label=j)
                    axs.fill_between(iters_to_plot, mus-sigmas, mus+sigmas, alpha=0.3, facecolor=model_to_colour_map[model_name])
                    axs.set_xscale('log')
                    axs.set_ylabel(metric_to_y_axis_lab[metric])
                    axs.set_xlabel('Iteration count')
                    plt.locator_params('x', len(iters_to_plot))
                    axs.title.set_text(pretty_dataset[dataset])

                fig.legend(*axs.get_legend_handles_labels(), loc='upper right')
                frmted_models = '-'.join([pretty_models[model] for model in models_to_plot])
                fig_name = os.path.join(plot_save_path, f'{pretty_dataset[dataset]}_{title_to_save_title[title]}_{prefix}_{frmted_models}_{metric_to_save_title[metric]}.pdf')
                print(f'Saving fig to...{fig_name}')
                fig.savefig(fname=fig_name, format='pdf')

def box_plot_downstream_model_performance_avr(results_by_dataset, title, models_to_plot, plot_save_path): 
    sns.set_style('darkgrid', {'grid.linestyle': ':'})
    metrics = ['avg_acc_n_samples_100+easy3_test',
                'avg_acc_n_samples_250+easy3_test',
                'avg_acc_n_samples_500+easy3_test',
                'avg_acc_n_samples_1000+easy3_test',
                'avg_acc_n_samples_10000+easy3_test',
                'avg_acc_n_samples_100000+easy3_test']
    plt.rc('axes', titlesize=22)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=28) 
    
    fig_e_1, axs_e_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.08, 
                            right=0.90,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.22,
                            hspace=0.10)
    fig_s_1, axs_s_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.08, 
                            right=0.90,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.22,
                            hspace=0.10)
    fig_e_2, axs_e_2 = plt.subplots(1, 3, figsize=(20, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.08, 
                            right=0.91,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.40,
                            hspace=0.10)
    fig_s_2, axs_s_2 = plt.subplots(1, 3, figsize=(20, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.08, 
                            right=0.91,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.40,
                            hspace=0.10)
    
    np.set_printoptions(precision=3, suppress=True)
    for i, metric in enumerate(metrics): 
        results_to_plot_standard, results_to_plot_embedded = get_results_to_plot(copy.deepcopy(results_by_dataset[SHAPES3D_DATASET]), metric)
        for axs, to_plot in zip([(axs_s_1, axs_s_2), (axs_e_1, axs_e_2)], [results_to_plot_standard, results_to_plot_embedded]): 
            to_plot = dict(map(
                lambda x: (x[0], list(set(x[1][200000]))), to_plot.items()
            ))
            to_plot = sorted(to_plot.items(), key=lambda x: SORT_ORDER[x[0]])
            to_plot = {k: np.array(v) for (k, v) in to_plot}
            for k in to_plot.keys(): 
                to_plot[k][to_plot[k] < 0] = 0
            if i < 2: 
                axs = axs[0]
                j = i 
            else: 
                axs = axs[1]
                j = i - 3
            plotted_vals = list(itertools.chain(*list(to_plot.values())))
            min_y = min(plotted_vals)
            max_y = max(plotted_vals)
            axs[j].set_ylim([min_y-0.05, max_y+0.05])
            sns.boxplot(data=to_plot, 
                        showfliers=False,
                        ax=axs[j], palette=model_to_colour_map, legend='brief')
            for model, vals in to_plot.items():
                print(f'****MODEL {model}, metric {metric}, dataset {dataset} ******')
                formatted_table_line = ''
                for (mu, sigma) in zip([np.array(vals).mean()], [np.array(vals).std()]): 
                    formatted_table_line += ' $\pm$ '.join([str(np.round(mu, 3)), str(np.round(sigma, 3))])
                    formatted_table_line += ' & '
                    print(formatted_table_line)
            axs[j].legend([], [], frameon=False)
            axs[j].set_xticklabels(str(i) for i in range(len(to_plot.keys())))
            axs[j].set_ylabel(metric_to_y_axis_lab[metric])
            axs[j].set_title(metric_to_title[metric])
            print(f'Order of plots {list(to_plot.keys())}')
            labels_s = axs_s_1[0].get_legend_handles_labels()[1]
            labels_e = axs_e_1[0].get_legend_handles_labels()[1]
            
    frmted_models = '-'.join([pretty_models[model] for model in models_to_plot])
    
    fig_name = os.path.join(plot_save_path, f'{title_to_save_title[title]}_standard_{frmted_models}_{metric_to_save_title[metric]}.pdf')
    print(f'Saving fig to...{fig_name}')
    fig_s_1.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
    fig_e_1.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
    fig_s_2.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
    fig_e_2.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
    fig_s_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_standard_pt-1_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
    fig_e_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_embedded_pt-1_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
    fig_s_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_standard_pt-2_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
    fig_e_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_embedded_pt-2_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')



def box_plot_downstream_model_performance(downstream_task: str, results_by_dataset, title=str, 
                                          models_to_plot=List[str], plot_save_path: str='/home/bethia/Documents/Code/Disentanglement/TPR/NewExp/baselines_schott/integrate_other_baselines/src/vis_results/plots/final') -> None: 
    if downstream_task == 'regression':
        metrics = ['reg_mlp/test_100_rsquared_excl_cat', 
                   'reg_mlp/test_250_rsquared_excl_cat', 
                   'reg_mlp/test_500_rsquared_excl_cat', 
                   'reg_mlp/test_1000_rsquared_excl_cat', 
                   'reg_mlp/test_10000_rsquared_excl_cat', 
                   'reg_mlp/test_full_rsquared_excl_cat']
    else: 
        box_plot_downstream_model_performance_avr(results_by_dataset, 
                                                  title, models_to_plot, plot_save_path)
        return 
        
    plt.rc('axes', titlesize=22)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=28) 
    sns.set_style('darkgrid', {'grid.linestyle': ':'})
    
    for dataset, results_dict in results_by_dataset.items(): 
        fig_e_1, axs_e_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.08, 
                                right=0.90,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.22,
                                hspace=0.10)
        fig_s_1, axs_s_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.08, 
                                right=0.90,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.22,
                                hspace=0.10)
        
        fig_e_2, axs_e_2 = plt.subplots(1, 3, figsize=(18, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.08, 
                                right=0.91,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.25,
                                hspace=0.10)
        fig_s_2, axs_s_2 = plt.subplots(1, 3, figsize=(18, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.08, 
                                right=0.91,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.25,
                                hspace=0.10)
        

        np.set_printoptions(precision=3, suppress=True)
        for i, metric in enumerate(metrics): 
            results_to_plot_standard, results_to_plot_embedded = get_results_to_plot(copy.deepcopy(results_dict), metric)
            for axs, to_plot in zip([axs_s_2, axs_e_2], [results_to_plot_standard, results_to_plot_embedded]): 
                to_plot = dict(map(
                    lambda x: (x[0], list(set(x[1][200000]))), to_plot.items()
                ))
                to_plot = sorted(to_plot.items(), key=lambda x: SORT_ORDER[x[0]])
                to_plot = {k: np.array(v) for (k, v) in to_plot}
                plotted_vals = list(itertools.chain(*list(to_plot.values())))
                min_y = min(plotted_vals)
                max_y = max(plotted_vals)
                if i < 2: 
                    continue 
                else: 
                    axs = axs[1]
                    j = i - 3
                axs_s_2[j].set_ylim([min_y-0.05, max_y+0.05])
                sns.boxplot(data=to_plot, 
                            showfliers=False,
                            ax=axs_s_2[j], palette=model_to_colour_map, legend='brief')
                for model, vals in to_plot.items():
                    print(f'****MODEL {model}, metric {metric}, dataset {dataset} ******')
                    formatted_table_line = ''
                    for (mu, sigma) in zip([np.array(vals).mean()], [np.array(vals).std()]): 
                        formatted_table_line += ' $\pm$ '.join([str(np.round(mu, 3)), str(np.round(sigma, 3))])
                        formatted_table_line += ' & '
                        print(formatted_table_line)
                axs_s_2[j].legend([], [], frameon=False)
                axs_s_2[j].set_xticklabels(str(i) for i in range(len(to_plot.keys())))
                axs_s_2[j].set_ylabel(metric_to_y_axis_lab[metric])
                axs_s_2[j].set_title(metric_to_title[metric])
                print(f'Order of plots {list(to_plot.keys())}')
                labels_s = axs_s_1[0].get_legend_handles_labels()[1]
                labels_e = axs_e_1[0].get_legend_handles_labels()[1]
            
            frmted_models = '-'.join([pretty_models[model] for model in models_to_plot])
            
            fig_name = os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}_standard_{frmted_models}_{metric_to_save_title[metric]}.pdf')
            print(f'Saving fig to...{fig_name}')
            #fig_s_1.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
            #fig_e_1.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
            #fig_s_2.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
            #fig_e_2.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
        
            fig_s_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}_standard_pt-1_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
            fig_e_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}_embedded_pt-1_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
            fig_s_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}_standard_pt-2_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
            fig_e_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}_embedded_pt-2_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')


# let's remove the models from sample efficiency if the final RSq is beneath a certain value 
# so we remove Shu from shapes3d
# we also remove shu and comet from cars3d
def box_plot_downstream_model_sample_eff_avr(results_by_dataset, title, models_to_plot, plot_save_path): 
    metrics = ['avg_acc_n_samples_100+easy3_test',
                'avg_acc_n_samples_250+easy3_test',
                'avg_acc_n_samples_500+easy3_test',
                'avg_acc_n_samples_1000+easy3_test',
                'avg_acc_n_samples_10000+easy3_test',
                'avg_acc_n_samples_100000+easy3_test']
    
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=25)
    
    
    sns.set_style('darkgrid', {'grid.linestyle': ':'})
    fig_e_1, axs_e_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.08, 
                            right=0.90,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.22,
                            hspace=0.10)
    #fig_e.suptitle(title, ha='center', va='bottom', y=0.05)
    fig_s_1, axs_s_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.06, 
                            right=0.90,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.22,
                            hspace=0.10)
    
    fig_e_2, axs_e_2 = plt.subplots(1, 3, figsize=(18, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.05, 
                            right=0.91,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.25,
                            hspace=0.10)
    #fig_e.suptitle(title, ha='center', va='bottom', y=0.05)
    fig_s_2, axs_s_2 = plt.subplots(1, 3, figsize=(18, 8)) # one plot per dataset
    plt.subplots_adjust(left=0.05, 
                            right=0.91,
                            bottom=0.1, 
                            top=0.9, 
                            wspace=0.25,
                            hspace=0.10)
    
    
    results_to_plot_standard_by_metric = {metric: {} for metric in metrics}
    results_to_plot_embedded_by_metric = {metric: {} for metric in metrics}
    
    for metric in metrics: 
        results_to_plot_standard, results_to_plot_embedded = get_results_to_plot(copy.deepcopy(results_by_dataset[SHAPES3D_DATASET]), metric)
        results_to_plot_standard = dict(map(
                lambda x: (x[0], np.array(list(set(x[1][200000])))), results_to_plot_standard.items()
            ))
        results_to_plot_embedded = dict(map(
                lambda x: (x[0], np.array(list(set(x[1][200000])))), results_to_plot_embedded.items()
            ))
        results_to_plot_standard_by_metric[metric] = results_to_plot_standard
        results_to_plot_embedded_by_metric[metric] = results_to_plot_embedded
        
    for metric in metrics: 
        for result_to_plot_by_metric in [results_to_plot_standard_by_metric, results_to_plot_embedded_by_metric]:
            for model, result_array in result_to_plot_by_metric[metric].items(): 
                min_items = min(result_array.shape[0], result_to_plot_by_metric[metrics[-1]][model].shape[0])
                sample_eff = result_array[:min_items]/result_to_plot_by_metric[metrics[-1]][model][:min_items]
                sample_eff[sample_eff > 1] = 1
                sample_eff[sample_eff < 0] = 0
                result_to_plot_by_metric[metric][model] = sample_eff
    np.set_printoptions(precision=3, suppress=True)
    for i, metric in enumerate(metrics[:-1]): 
        results_to_plot_standard = results_to_plot_standard_by_metric[metric]
        results_to_plot_embedded = results_to_plot_embedded_by_metric[metric]
        for axs, to_plot in zip([(axs_s_1, axs_s_2), (axs_e_1, axs_e_2)], [results_to_plot_standard, results_to_plot_embedded]): 
            to_plot = sorted(to_plot.items(), key=lambda x: SORT_ORDER[x[0]])
            to_plot = {k: v for (k, v) in to_plot}
            plotted_vals = list(itertools.chain(*list(to_plot.values())))
            min_y = min(plotted_vals)
            max_y = max(plotted_vals)
            if i < 2: 
                axs = axs[0]
                j = i 
            else: 
                axs = axs[1]
                j = i - 2
            axs[j].set_ylim([min_y-0.05, max_y+0.05])
            sns.boxplot(data=to_plot, 
                        showfliers=False,
                        ax=axs[j],
                        palette=model_to_colour_map,
                        legend='brief')
            for model, vals in to_plot.items():
                formatted_table_line = ''
                for (mu, sigma) in zip([np.array(vals).mean()], [np.array(vals).std()]): 
                    formatted_table_line += ' $\pm$ '.join([str(np.round(mu, 3)), str(np.round(sigma, 3))])
                    formatted_table_line += ' & '
            axs[j].set_xticklabels(str(i) for i in range(len(to_plot.keys())))
            axs[j].set_ylabel('Accuracy ratio')
            axs[j].legend([], [], frameon=False)
            axs[j].set_title(sample_eff_to_title[metric])
            
    frmted_models = '-'.join([pretty_models[model] for model in models_to_plot])
    labels_s = axs_s_1[0].get_legend_handles_labels()[1]
    labels_e = axs_e_1[0].get_legend_handles_labels()[1]
    fig_s_1.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
    fig_e_1.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
    fig_s_2.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
    fig_e_2.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
    fig_s_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_avr_standard_pt-1_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
    fig_e_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_avr_embedded_pt-1_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
    fig_s_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_avr_standard_pt-2_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
    fig_e_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_avr_embedded_pt-2_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
    

def box_plot_downstream_model_sample_eff(downstream_task, results_by_dataset, title=str, 
                                          models_to_plot=List[str], plot_save_path: str='/home/bethia/Documents/Code/Disentanglement/TPR/NewExp/baselines_schott/integrate_other_baselines/src/vis_results/plots/final') -> None: 
    if downstream_task != 'regression':
        box_plot_downstream_model_sample_eff_avr(results_by_dataset, title, 
                                                 models_to_plot, plot_save_path)
        return 
    
    plt.rc('axes', titlesize=24)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=26) 
    
    metrics = ['reg_mlp/test_100_rsquared_excl_cat', 
               'reg_mlp/test_250_rsquared_excl_cat', 
               'reg_mlp/test_500_rsquared_excl_cat', 
               'reg_mlp/test_1000_rsquared_excl_cat', 
               'reg_mlp/test_10000_rsquared_excl_cat', 
               'reg_mlp/test_full_rsquared_excl_cat']
    sns.set_style('darkgrid', {'grid.linestyle': ':'})
    for dataset, results_dict in results_by_dataset.items(): 
        fig_e_1, axs_e_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.06, 
                                right=0.90,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.22,
                                hspace=0.10)
        fig_s_1, axs_s_1 = plt.subplots(1, 2, figsize=(15, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.06, 
                                right=0.90,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.22,
                                hspace=0.10)
        fig_e_2, axs_e_2 = plt.subplots(1, 3, figsize=(18, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.05, 
                                right=0.91,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.25,
                                hspace=0.10)
        fig_s_2, axs_s_2 = plt.subplots(1, 3, figsize=(18, 8)) # one plot per dataset
        plt.subplots_adjust(left=0.05, 
                                right=0.91,
                                bottom=0.1, 
                                top=0.9, 
                                wspace=0.25,
                                hspace=0.10)
            
        results_to_plot_standard_by_metric = {metric: {} for metric in metrics}
        results_to_plot_embedded_by_metric = {metric: {} for metric in metrics}
        
        for metric in metrics: 
            results_to_plot_standard, results_to_plot_embedded = get_results_to_plot(copy.deepcopy(results_dict), metric)
            results_to_plot_standard = dict(map(
                    lambda x: (x[0], np.array(list(set(x[1][200000])))), results_to_plot_standard.items()
                ))
            results_to_plot_embedded = dict(map(
                    lambda x: (x[0], np.array(list(set(x[1][200000])))), results_to_plot_embedded.items()
                ))
            results_to_plot_standard_by_metric[metric] = results_to_plot_standard
            results_to_plot_embedded_by_metric[metric] = results_to_plot_embedded
            
        for metric in metrics: 
            for result_to_plot_by_metric in [results_to_plot_standard_by_metric, results_to_plot_embedded_by_metric]:
                for model, result_array in result_to_plot_by_metric[metric].items(): 
                    min_items = min(result_array.shape[0], result_to_plot_by_metric[metrics[-1]][model].shape[0])
                    sample_eff = result_array[:min_items]/result_to_plot_by_metric[metrics[-1]][model][:min_items]
                    sample_eff[sample_eff > 1] = 1
                    sample_eff[sample_eff < 0] = 0
                    result_to_plot_by_metric[metric][model] = sample_eff

        np.set_printoptions(precision=3, suppress=True)
        for i, metric in enumerate(metrics[:-1]): 
            results_to_plot_standard = results_to_plot_standard_by_metric[metric]
            results_to_plot_embedded = results_to_plot_embedded_by_metric[metric]
            for axs, to_plot in zip([(axs_s_1, axs_s_2), (axs_e_1, axs_e_2)], [results_to_plot_standard, results_to_plot_embedded]): 
                to_plot = sorted(to_plot.items(), key=lambda x: SORT_ORDER[x[0]])
                to_plot = {k: v for (k, v) in to_plot}
                plotted_vals = list(itertools.chain(*list(to_plot.values())))
                min_y = min(plotted_vals)
                max_y = max(plotted_vals)
                if i < 2: 
                    axs = axs[0]
                    j = i 
                else: 
                    axs = axs[1]
                    j = i - 2
                axs[j].set_ylim([min_y-0.05, max_y+0.05])
                for model, vals in to_plot.items():
                    print(f'****MODEL {model}, metric {metric}, dataset {dataset} ******')
                    formatted_table_line = ''
                    for (mu, sigma) in zip([np.array(vals).mean()], [np.array(vals).std()]): 
                        formatted_table_line += ' $\pm$ '.join([str(np.round(mu, 3)), str(np.round(sigma, 3))])
                        formatted_table_line += ' & '
                        print(formatted_table_line)
                if 'shapes' in dataset: 
                    try:
                        to_plot.pop(SHU)
                        to_plot.pop(f'{SHU}_embed')
                        print(f'Removed shu')
                    except: 
                        pass 
                if 'cars' in dataset: 
                    try:
                        to_plot.pop(SHU)
                        to_plot.pop(COMET)
                        to_plot.pop(f'{SHU}_embed')
                        to_plot.pop(f'{COMET}_embed')
                        print(f'Removed comet, shu')
                    except:
                        pass
                g = sns.boxplot(data=to_plot, 
                            showfliers=False,
                            ax=axs[j],
                            palette=model_to_colour_map,
                            legend='brief')
                axs[j].legend([], [], frameon=False)
                axs[j].set_xticklabels(str(i) for i in range(len(to_plot.keys())))
                print(f'Order of plots {list(to_plot.keys())}')
                axs[j].set_ylabel('Rsq ratio')
                axs[j].set_title(sample_eff_to_title[metric])
        frmted_models = '-'.join([pretty_models[model] for model in models_to_plot])
        labels_s = axs_s_1[0].get_legend_handles_labels()[1]
        labels_e = axs_e_1[0].get_legend_handles_labels()[1]
        print(labels_s)
        fig_s_1.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
        fig_e_1.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
        fig_s_2.legend(*(axs_s_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_s)))), loc='upper right')
        fig_e_2.legend(*(axs_e_1[0].get_legend_handles_labels()[0], list(str(i) for i in range(len(labels_e)))), loc='upper right')
        fig_s_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}-pt_1_standard_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
        fig_e_1.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}-pt_1_embedded_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
        fig_s_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}-pt_2_standard_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')
        fig_e_2.savefig(fname=os.path.join(plot_save_path, f'{title_to_save_title[title]}_{dataset}-pt_2_embedded_{frmted_models}_{metric_to_save_title[metric]}.pdf'), format='pdf')



R_SQUARED = 'rsq'
SAMPLE_EFFICIENCY = 'sample_eff'
DIS_BETA = 'dis_beta'
DIS_FAC = 'dis_fac'
DIS_DCI = 'dis_dci'
DIS_MIG = 'dis_mig'

# shapes downstream 100 (particularly), 250, 500 samples


mode_to_title = {'repn_learner_convergence_downstream': 'Convergence rate of representation learner (downstream)', 
                 'repn_learner_convergence_dis_metrics': 'Convergence rate of representation learner (disentanglement)',
                 'downstream_performance': 'Performance of downstream model',
                 'downstream_sample_eff': 'Sample efficiency of downstream model'}

title_to_save_title = {'Convergence rate of representation learner (downstream)': 'cr_repn_learner_downstream',
                       'Convergence rate of representation learner (disentanglement)': 'cr_repn_learner_dis_metrics',
                       'Performance of downstream model': 'downstream_perf',
                       'Sample efficiency of downstream model': 'sample_eff'}
metric_to_save_title = {
                'reg_mlp/test_full_rsquared_excl_cat': 'rsq_full',
                'reg_mlp/test_10000_rsquared_excl_cat': 'rsq_10000',
                'reg_mlp/test_1000_rsquared_excl_cat': 'rsq_1000',
                'reg_mlp/test_100_rsquared_excl_cat': 'rsq_100',
                'reg_mlp/test_250_rsquared_excl_cat': 'rsq_250',
                'reg_mlp/test_500_rsquared_excl_cat': 'rsq_500',
                'reg_mlp/diff/_100_final_rsq_exc/10000_final_rsq_exc': 'se_100_10000',
                'reg_mlp/diff/_1000_final_rsq_exc/10000_final_rsq_exc':'se_1000_10000',
                'reg_mlp/diff/_100_final_rsq_exc/full_final_rsq_exc': 'se_100_full',
                'reg_mlp/diff/_1000_final_rsq_exc/full_final_rsq_exc': 'se_1000_full',
                'reg_mlp/diff/_10000_final_rsq_exc/full_final_rsq_exc': 'se_10000_full',
                'reg_mlp/diff/_500_final_rsq_exc/10000_final_rsq_exc': 'se_500_10000',
                'reg_mlp/diff/_500_final_rsq_exc/full_final_rsq_exc': 'se_500_full',
                'reg_mlp/diff/_250_final_rsq_exc/10000_final_rsq_exc': 'se_250_10000',
                'reg_mlp/diff/_250_final_rsq_exc/full_final_rsq_exc': 'se_250_full',
                'dmetric/train/_beta_val_acc': 'beta_score',
                'dmetric/train/_fac_eval': 'fac_score',
                'dmetric/train/_discrete_mig': 'mig_score',
                'dmetric/train/_dci_disentanglement': 'dci_score',
                'avg_acc_n_samples_100000+easy3_test': 'acc_100000',
                'avg_acc_n_samples_10000+easy3_test': 'acc_10000',
                'avg_acc_n_samples_1000+easy3_test': 'acc_1000',
                'avg_acc_n_samples_500+easy3_test': 'acc_500',
                'avg_acc_n_samples_250+easy3_test': 'acc_250',
                'avg_acc_n_samples_100+easy3_test': 'acc_100',
}

pretty_models ={ 
        'gvae': 'g',
        'gvae_embed': 'g_e',
        'adagvae': 'a',
        'adagvae_embed': 'a_e', 
        'mlvae': 'm',
        'mlvae_embed': 'm_e',
        'slowvae': 's', 
        'slowvae_embed': 's_e', 
        'shu': 'sh', 
        'shu_embed': 'sh_e', 
        'comet': 'co',
        'comet_embed': 'co_e',
        'vct': 'vct',
        'vct_embed': 'vct_e',
        'new_ae': 'new_ae',
        'new_ae_filler_idxs': 'fidx',
        'new_ae_quantised_fillers_concatenated': 'bf', 
        'new_ae_soft_fillers_concatenated': 'af', 
        'new_ae_z_tpr': 'tpr',
        'new_ae_z_soft_tpr': 'soft_tpr'

}

metric_to_title = {
    'reg_mlp/test_full_rsquared_excl_cat': 'Downstream performance (all)',
                'reg_mlp/test_10000_rsquared_excl_cat': 'Downstream performance (10,000)',
                'reg_mlp/test_1000_rsquared_excl_cat': 'Downstream performance (1,000)',
                'reg_mlp/test_100_rsquared_excl_cat': 'Downstream performance (100)',
                'reg_mlp/test_250_rsquared_excl_cat': 'Downstream performance (250)',
                'reg_mlp/test_500_rsquared_excl_cat': 'Downstream performance (500)',
                'avg_acc_n_samples_100000+easy3_test': 'Downstream performance (100,000)',
                'avg_acc_n_samples_10000+easy3_test': 'Downstream performance (10,000)',
                'avg_acc_n_samples_1000+easy3_test': 'Downstream performance (1,000)',
                'avg_acc_n_samples_500+easy3_test': 'Downstream performance (500)',
                'avg_acc_n_samples_250+easy3_test': 'Downstream performance (250)',
                'avg_acc_n_samples_100+easy3_test': 'Downstream performance (100)',
}

sample_eff_to_title = {
                'reg_mlp/test_10000_rsquared_excl_cat': 'Sample efficiency (10,000/all)',
                'reg_mlp/test_1000_rsquared_excl_cat': 'Sample efficiency (1,000/all)',
                'reg_mlp/test_100_rsquared_excl_cat': 'Sample efficiency (100/all)',
                'reg_mlp/test_250_rsquared_excl_cat': 'Sample efficiency (250/all)',
                'reg_mlp/test_500_rsquared_excl_cat': 'Sample efficiency (500/all)',
                'avg_acc_n_samples_10000+easy3_test': 'Sample efficiency (10,000/100,000)',
                'avg_acc_n_samples_1000+easy3_test': 'Sample efficiency (1,000/100,000)',
                'avg_acc_n_samples_500+easy3_test': 'Sample efficiency (500/100,000)',
                'avg_acc_n_samples_250+easy3_test': 'Sample efficiency (250/100,000)',
                'avg_acc_n_samples_100+easy3_test': 'Sample efficiency (100/100,000)',
}

pretty_dataset = {
    'cars3d': 'Cars3D',
    'shapes3d': 'Shapes3D',
    'mpi3d': 'MPI3D'
}

metric_to_y_axis_lab = {'reg_mlp/test_full_rsquared_excl_cat': 'RSq',
                'reg_mlp/test_10000_rsquared_excl_cat': 'RSq',
                'reg_mlp/test_1000_rsquared_excl_cat': 'RSq',
                'reg_mlp/test_100_rsquared_excl_cat': 'RSq',
                'reg_mlp/test_250_rsquared_excl_cat': 'RSq',
                'reg_mlp/test_500_rsquared_excl_cat': 'RSq',
                'reg_mlp/diff/_100_final_rsq_exc/10000_final_rsq_exc': 'RSq ratio',
                'reg_mlp/diff/_1000_final_rsq_exc/10000_final_rsq_exc':'RSq ratio',
                'reg_mlp/diff/_100_final_rsq_exc/full_final_rsq_exc': 'Rsq ratio',
                'reg_mlp/diff/_1000_final_rsq_exc/full_final_rsq_exc': 'Rsq ratio',
                'reg_mlp/diff/_10000_final_rsq_exc/full_final_rsq_exc': 'Rsq ratio',
                'reg_mlp/diff/_500_final_rsq_exc/10000_final_rsq_exc': 'Rsq ratio',
                'reg_mlp/diff/_500_final_rsq_exc/full_final_rsq_exc': 'Rsq ratio',
                'reg_mlp/diff/_250_final_rsq_exc/10000_final_rsq_exc': 'Rsq ratio',
                'reg_mlp/diff/_250_final_rsq_exc/full_final_rsq_exc': 'Rsq ratio',

                'dmetric/train/_beta_val_acc': 'Beta score',
                'dmetric/train/_fac_eval': 'Factor score',
                'dmetric/train/_discrete_mig': 'MIG score',
                'dmetric/train/_dci_disentanglement': 'DCI score',

                'avg_acc_n_samples_100000+easy3_test': 'Accuracy',
                'avg_acc_n_samples_10000+easy3_test': 'Accuracy',
                'avg_acc_n_samples_1000+easy3_test': 'Accuracy',
                'avg_acc_n_samples_500+easy3_test': 'Accuracy',
                'avg_acc_n_samples_250+easy3_test': 'Accuracy',
                'avg_acc_n_samples_100+easy3_test': 'Accuracy',
                }

SORT_ORDER = {
        'slowvae': 0, 
        'slowvae_embed': 1, 
        'adagvae': 2,
        'adagvae_embed': 3, 
        'gvae': 4,
        'gvae_embed': 5,
        'mlvae': 6,
        'mlvae_embed': 7,
        'shu': 8,
        'shu_embed': 9,
        'vct': 10, 
        'vct_embed': 11,
        'comet': 12,
        'comet_embed': 13, 
        'new_ae_filler_idxs': 14,
        'new_ae_quantised_fillers_concatenated': 15, 
        'new_ae_soft_fillers_concatenated': 16, 
        'new_ae_z_tpr': 17,
        'new_ae_z_soft_tpr': 18,
        'new_ae': 19,
}


clrs = sns.color_palette('deep')
model_to_colour_map = {
    'gvae': clrs[7],
    'gvae_embed': clrs[7],
    'adagvae': clrs[1],
    'adagvae_embed': clrs[1], 
    'mlvae': clrs[2],
    'mlvae_embed': clrs[2],
    'slowvae': clrs[3], 
    'slowvae_embed': clrs[3], 
    'shu': clrs[4],
    'shu_embed': clrs[4],
    'comet': clrs[5],
    'comet_embed': clrs[5],
    'vct': clrs[6], 
    'vct_embed': clrs[6], 
    'new_ae_filler_idxs': clrs[1],
    'new_ae_quantised_fillers_concatenated': clrs[2], 
    'new_ae_soft_fillers_concatenated': clrs[0], 
    'new_ae_z_tpr': clrs[8],
    'new_ae_z_soft_tpr': clrs[0],
    'new_ae': clrs[0],
}

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='sample_eff', choices=['downstream_performance', 
                                                             'downstream_sample_eff', 
                                                             'repn_learner_convergence_downstream',
                                                             'repn_learner_convergence_dis_metrics'])
parser.add_argument('--downstream_task', default='regression', choices=['regression', 'avr'])
parser.add_argument('--dataset', default=SHAPES3D_DATASET, choices=DATASET_CHOICES)
parser.add_argument('--models_to_plot', default='gvae,adagvae,mlvae,slowvae,comet,vct,shu,new_ae')
parser.add_argument('--soft_tpr_variants_to_plot', default='new_ae_z_soft_tpr,') # specify which soft tprs we want to plot

if __name__ == '__main__': 

    args = parser.parse_args()
    ## SHAPES3D 
    model_type_to_load_dir_parent_map = {
        SHAPES3D_DATASET: {
            SOFT_TPR_AE: ['/media/bethia/F6D2E647D2E60C25/trained/own_model/3dshapes/final_saved_every_1000_iters/',
                            '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/3dshapes/final_saved_every_1000_iters/',
                            '/g/data/po67/anonymous_cat/Trained/own_model/shapes3d'],
            GVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/gvae_TUNED/', 
                '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/shapes3d/gvae_TUNED/',
                '/g/data/po67/anonymous_cat/Trained/baselines/shapes3d/gvae'],
            ADAGVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/adagvae_k_known-FINAL/',
                    '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/shapes3d/adagvae_k_known-FINAL',
                    '/g/data/po67/anonymous_cat/Trained/baselines/shapes3d/adagvae_k_known'],
            MLVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/mlvae_TUNED/',
                    '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/shapes3d/mlvae_TUNED/',
                    '/g/data/po67/anonymous_cat/Trained/baselines/shapes3d/mlvae'],
            SLOWVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/shapes3d/slowvae/',
                    '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/shapes3d/slowvae/',
                    '/g/data/po67/anonymous_cat/Trained/baselines/shapes3d/slowvae'], 
            COMET: ['/g/data/po67/anonymous_cat/Trained/COMET/shapes3d'], 
            VCT: ['/g/data/po67/anonymous_cat/Trained/VCT/shapes3d'],
            SHU: ['/g/data/po67/anonymous_cat/Trained/baselines/shapes3d/shu',
                  '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/shapes3d/shu-FINAL-lower_dis_lr'],
        } 
        ,
        MPI3D_DATASET: {
            SOFT_TPR_AE: ['/media/bethia/F6D2E647D2E60C25/trained/own_model/mpi3d/',
                            '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/mpi3d/mpi3d/',
                            '/g/data/po67/anonymous_cat/Trained/own_model/mpi3d'],
            GVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/gvae-FINAL/', 
                '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/mpi3d/gvae-FINAL/',
                '/g/data/po67/anonymous_cat/Trained/baselines/mpi3d/gvae'],
            ADAGVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/adagvae_k_known-FINAL/',
                    '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/mpi3d/adagvae_k_known-FINAL',
                    '/g/data/po67/anonymous_cat/Trained/baselines/mpi3d/adagvae_k_known'],
            MLVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/mlvae-FINAL/',
                    '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/mpi3d/mlvae-FINAL/',
                    '/g/data/po67/anonymous_cat/Trained/baselines/mpi3d/mlvae'],
            SLOWVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/mpi3d/slowvae-FINAL/',
                    '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/mpi3d/slowvae-FINAL/',
                    '/g/data/po67/anonymous_cat/Trained/baselines/mpi3d/slowvae'],
            COMET: ['/g/data/po67/anonymous_cat/Trained/COMET/mpi3d'], 
            VCT: ['/g/data/po67/anonymous_cat/Trained/VCT/mpi3d'],
            SHU: ['/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/mpi3d/shu-FINAL-lower_dis_lr',
                  '/g/data/po67/anonymous_cat/Trained/baselines/mpi3d/shu'],
        },
        CARS3D_DATASET: {
            SOFT_TPR_AE: ['/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/own_model/cars3d',
                              '/media/bethia/F6D2E647D2E60C25/trained/own_model/cars3d',
                              '/g/data/po67/anonymous_cat/Trained/own_model/cars3d'],
            GVAE: ['/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/gvae-FINAL',
                   '/media/bethia/F6D2E647D2E60C25/trained/baselines/cars3d/gvae-FINAL',
                   '/g/data/po67/anonymous_cat/Trained/baselines/cars3d/gvae'],
            ADAGVAE:['/media/bethia/F6D2E647D2E60C25/trained/baselines/cars3d/adagvae_k_known-FINAL', 
                     '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/adagvae_k_known-FINAL',
                     '/g/data/po67/anonymous_cat/Trained/baselines/cars3d/adagvae_k_known'], 
            MLVAE: ['/media/bethia/F6D2E647D2E60C25/trained/baselines/cars3d/mlvae-FINAL',
                    '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/mlvae-FINAL',
                    '/g/data/po67/anonymous_cat/Trained/baselines/cars3d/mlvae'],
            SLOWVAE: ['/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/slowvae-FINAL',
                      '/media/bethia/F6D2E647D2E60C25/trained/baselines/cars3d/slowvae-FINAL',
                      '/g/data/po67/anonymous_cat/Trained/baselines/cars3d/slowvae'], 
            COMET: ['/g/data/po67/anonymous_cat/Trained/COMET/cars'], 
            VCT: ['/g/data/po67/anonymous_cat/Trained/VCT/cars3d'],
            SHU: ['/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/cars3d/shu-FINAL',
                  '/g/data/po67/anonymous_cat/Trained/baselines/cars3d/shu'],
        }
    }

    wandb_log_path = '/home/bethia/Documents/Code/Disentanglement/TPR/NewExp/baselines_schott/integrate_other_baselines/wandb/'
    mode = args.mode
    models_to_plot = [model for model in args.models_to_plot.split(',')]

    soft_tpr_variants_to_plot = [tpr_variant for tpr_variant in args.soft_tpr_variants_to_plot.split(',')]
        
    if args.mode == 'downstream_performance' or args.mode == 'downstream_sample_eff': 
        model_iters_to_plot = [200000]
    else: 
        model_iters_to_plot = [100, 1000, 10000, 100000, 200000]
    
    results_by_dataset = {dataset: {} for dataset in DATASET_CHOICES}
    for dataset in [SHAPES3D_DATASET, CARS3D_DATASET]: 
        results_by_dataset[dataset]=parse_results_from_wandb_logs(mode, 
                                             model_type_to_load_dir_parent_map[dataset], 
                                             iters_to_plot=model_iters_to_plot,
                                             models_to_plot=models_to_plot,
                                             soft_tpr_variants_to_plot=soft_tpr_variants_to_plot,
                                             wandb_log_path=wandb_log_path)
    
    
    if args.mode == 'downstream_performance': 
        box_plot_downstream_model_performance(args.downstream_task, 
                                              results_by_dataset, title=mode_to_title[args.mode],
                                              models_to_plot=models_to_plot)
    if args.mode == 'downstream_sample_eff': 
        box_plot_downstream_model_sample_eff(args.downstream_task, 
                                             results_by_dataset, 
                                             title=mode_to_title[args.mode],
                                             models_to_plot=models_to_plot)
    if args.mode == 'downstream_convergence':
        raise NotImplementedError
    if args.mode in ['repn_learner_convergence_downstream', 'repn_learner_convergence_dis_metrics']:
        line_plot_repn_learner_convergence(args.downstream_task, results_by_dataset, mode=args.mode, 
                                           models_to_plot=models_to_plot, title=mode_to_title[args.mode])



   