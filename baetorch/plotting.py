import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from .evaluation import calc_mean_results, calc_fpr80
import scipy.stats as stats
import torch
from .util.misc import get_sample_dataloader, create_dir

#modularised plots and evaluation
def plot_samples_img(data=[],plot_samples=10, reshape_size=None, savefile="", savefolder="plots"):
    if isinstance(data,dict):
        data_values = list(data.values())
        data_names = list(data.keys())
    else:
        data_values = data
        data_names = [str(i) for i in range(1,len(data_values)+1)]

    num_types = len(data_values)
    fig, ax = plt.subplots(plot_samples,num_types, gridspec_kw={'wspace':0, 'hspace':0},squeeze=True,figsize=(9,6))

    for plot_sample in range(plot_samples):
        for model_output,data_name in zip(np.arange(num_types),data_names):

            image = data_values[model_output][plot_sample]

            #reshape images if necessary
            if reshape_size is not None:
                if len(reshape_size) ==3: #RGB channels
                    dim = reshape_size[-1]
                    image_dim = reshape_size[:-1]
                    image = np.reshape(image, (dim,image_dim[0],image_dim[1]))
                    image = np.moveaxis(image, 0, -1)
                else:
                    image = np.reshape(image, reshape_size)

            if len(reshape_size) ==3:

                if data_name != "input" and data_name != "mu":
                    image = image.mean(-1)

            ax[plot_sample][model_output].axis("off")
            ax[plot_sample][model_output].imshow(image, cmap='viridis', aspect='auto')
            ax[0][model_output].set_title(data_names[model_output],fontsize=6)
    plt.subplots_adjust(hspace=0, wspace=0)

    #option to save plot
    if '.png' in savefile:
        create_dir(savefolder)
        plt.savefig(savefolder+"/"+savefile)

    return fig

def plot_output_distribution(*args, legends=["TEST", "OOD"], exclude_keys=[], savefile="", savefolder="plots"):
    #exclude mu and input
    exclude_keys.append("mu")
    exclude_keys.append("input")

    num_datasets = len(args)
    metrics_mean, axis_titles = calc_mean_results(args[0],exclude_keys=exclude_keys) #redundant calculation, but it works. idea is to get the metric titles
    num_metrics = len(axis_titles)
    print(axis_titles)
    #plot histogram of nll, etc
    figsize = (15,15)
    fig, axes = plt.subplots(num_metrics,2,figsize=figsize)
    dataset_metrics = []
    for predict_results in args:
        metrics_mean, axis_titles = calc_mean_results(predict_results,exclude_keys=exclude_keys)
        dataset_metrics.append(metrics_mean)
        normed = True

        #histograms
        alpha= 0.7
        for i,metric in enumerate(metrics_mean):
            axes[i][0].hist(metric,density=normed,alpha=alpha)

    #box plots
    for metric in range(num_metrics):
        dataset_metric = [dataset_metrics[num_dt][metric] for num_dt in range(num_datasets)]
        axes[metric][1].boxplot(dataset_metric)
        axes[metric][1].set_xticklabels(legends)

        #calculate and set medians
        medians_title = ""
        for num_dt in range(num_datasets):
            medians_title += (" "+legends[num_dt] +'(%0.2f)' % np.median(dataset_metric[num_dt]))
        axes[metric][1].set_title(medians_title)

    #set title for first column (histograms)
    for ax,title in zip(axes[:,0],axis_titles):
        ax.legend(legends)
        ax.set_title(title)

    #option to save plot
    if '.png' in savefile:
        create_dir(savefolder)
        plt.savefig(savefolder+"/"+savefile)

#plot ROC curves and return AUROC, FPR80
def plot_roc_curve(results_test,results_ood, title="OOD", exclude_keys=[],savefile="",savefolder="plots"):
    #exclude mu and input
    exclude_keys.append("mu")
    exclude_keys.append("input")

    plt.figure()
    lw = 2
    test_scores,metric_names = calc_mean_results(results_test,exclude_keys=exclude_keys)
    ood_scores,metric_names = calc_mean_results(results_ood,exclude_keys=exclude_keys)

    fpr80_list = []
    auroc_list = []

    for metric_num, metric_name in enumerate(metric_names):
        test_score = test_scores[metric_num]
        ood_score = ood_scores[metric_num]

        total_scores = np.concatenate((test_score,ood_score))
        y_true = np.concatenate((np.zeros_like(test_score),np.ones_like(ood_score)))
        fpr, tpr, thresholds = metrics.roc_curve(y_true, total_scores, pos_label=1)
        auroc_score = metrics.roc_auc_score(y_true, total_scores)
        fpr80 = calc_fpr80(fpr,tpr)
        fpr80_list.append(np.round(fpr80,4))
        auroc_list.append(np.round(auroc_score,4))
        plt.plot(fpr, tpr,
                 lw=lw, label=metric_name+'(area = %0.4f)' % auroc_score)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) for '+title+' metrics')
    plt.legend(loc="lower right")

    #option to save plot
    if '.png' in savefile:
        create_dir(savefolder)
        plt.savefig(savefolder+"/"+savefile)

    return auroc_list, fpr80_list, metric_names

#plot precision-recall curve and calculate AUPRC
def plot_prc_curve(results_test,results_ood,title="OOD", exclude_keys=[],savefile="",savefolder="plots"):

    #exclude mu and input
    exclude_keys.append("mu")
    exclude_keys.append("input")

    plt.figure()
    lw = 2
    test_scores,metric_names = calc_mean_results(results_test,exclude_keys=exclude_keys)
    ood_scores,metric_names = calc_mean_results(results_ood,exclude_keys=exclude_keys)

    auprc_list = []
    for metric_num, metric_name in enumerate(metric_names):
        test_score = test_scores[metric_num]
        ood_score = ood_scores[metric_num]

        total_scores = np.concatenate((test_score,ood_score))
        y_true = np.concatenate((np.zeros_like(test_score),np.ones_like(ood_score)))
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, total_scores, pos_label=1)
        auprc_score = metrics.auc(recall, precision)
        auprc_list.append(np.round(auprc_score,4))
        plt.plot(recall, precision,
                 lw=lw, label=metric_name+'(area = %0.4f)' % auprc_score)

    plt.plot([0, 1], [0.5, 0.5], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.48, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC) for '+title+' metrics')
    plt.legend(loc="lower right")

    #option to save plot
    if '.png' in savefile:
        create_dir(savefolder)
        plt.savefig(savefolder+"/"+savefile)

    return auprc_list, metric_names

def plot_calibration_curve(*models, id_data_test=None,savefile="",savefolder="plots"):
    #allow flexible in handling arguments list
    if id_data_test is None and len(models) >= 2:
        id_data_test = models[-1]
        models = models[:-1]

    plt.figure()
    for model in models:
        predict_test = model.predict(id_data_test)

        #collect statistics
        stds = {}
        std_epistemic = predict_test['epistemic']
        stds.update({"epistemic":std_epistemic})

        #check conditions for homoscedestic noise
        if model.homoscedestic_mode != "none":
            if model.homoscedestic_mode == "every":
                std_homo_alea = predict_test['epistemic'] + np.array([model.get_homoscedestic_noise(return_mean=False)[0]]*predict_test['epistemic'].shape[0])
            elif model.homoscedestic_mode == "single":
                std_homo_alea = predict_test['epistemic'] + model.get_homoscedestic_noise()
            stds.update({"homo_aleatoric":std_homo_alea})

        #check if decoder sigma is enabled
        if model.decoder_sigma_enabled:
            std_hetero_alea = predict_test['aleatoric']
            std_epi_hetero_alea = predict_test['epistemic']+predict_test['aleatoric']
            stds.update({"hetero_aleatoric":std_hetero_alea})
            stds.update({"epi_hetero_aleatoric":std_epi_hetero_alea})

        #get surd of variance for each metric
        stds = {key:np.sqrt(val) for key,val in stds.items()}

        #start plotting
        mean_calibration_errors = []
        legends = []

        for key, std in stds.items():

            mu = predict_test['mu'].reshape(-1)
            std = std.reshape(-1)
            y_true = predict_test['input'].reshape(-1)
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            total_data_points = len(mu)
            print("------------------")
            credible_interval_range = [0.1, 0.3,0.5, 0.7, 0.9,0.95, 0.99]
            observed_perc_list = []
            for credible_interval in credible_interval_range:
                z_score = stats.norm.ppf(1-(1-credible_interval)/2)

                lower_bound = np.clip(mu-z_score*std,a_min=0,a_max=1.)
                upper_bound = mu+z_score*std

                num_within = np.argwhere((y_true>=lower_bound) & (y_true<=upper_bound)).shape[0]

                observed_perc = num_within/total_data_points
                observed_perc_list.append(observed_perc)

            mean_calibration_error = np.mean(np.abs(np.array(observed_perc_list)-np.array(credible_interval_range)))
            mean_calibration_errors.append(mean_calibration_error)
            print(model.model_name+" "+key+" MCE="+str(mean_calibration_error))
            legends.append(model.model_name+" "+key+" CE=(%0.2f)"% mean_calibration_error)
            ideal_xrange = np.arange(0.0,1.1,0.1)
            plt.plot(credible_interval_range,observed_perc_list)

    #option to save plot
    if '.png' in savefile:
        create_dir(savefolder)
        plt.savefig(savefolder+"/"+savefile)

    plt.legend(legends)
    plt.plot(ideal_xrange,ideal_xrange,'-.')

def plot_latent(*datasets, legends=[],bae_model, transform_pca=True, return_var=True, alpha=0.9):
    #return var validity depends on bae model.num_samples
    if return_var and bae_model.num_samples >1:
        return_var = True
    else:
        return_var = False

    #do actual plot
    if return_var:
        fig, (ax_mu,ax_sig) = plt.subplots(1,2)
    else:
        fig, (ax_mu) = plt.subplots(1,1)
    for dataset in datasets:
        data_latent_mu,data_latent_sig = bae_model.predict_latent(get_sample_dataloader(dataset)[0],transform_pca)
        ax_mu.scatter(data_latent_mu[:,0],data_latent_mu[:,1],alpha=alpha)
        if return_var :
           ax_sig.scatter(data_latent_sig[:,0],data_latent_sig[:,1],alpha=alpha)

    if len(legends) >0:
        ax_mu.legend(legends)
        if return_var:
            ax_sig.legend(legends)

    #set titles
    ax_mu.set_title(bae_model.model_name+" latent mean")
    if return_var:
        ax_sig.set_title(bae_model.model_name+" latent variance")


def plot_train_loss(model,savefile="",savefolder="plots"):
    plt.figure()
    losses = model.losses
    if len(model.losses)>=5:
        losses = losses[5:]

    plt.plot(np.arange(1,len(losses)+1),losses)
    plt.title(model.model_name+" Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    #option to save plot
    if '.png' in savefile:
        create_dir(savefolder)
        plt.savefig(savefolder+"/"+savefile)

def get_grid2d_latent(latent_data, span=1):
    """
    Returns a grid for plotting contours by providing the reference 2D data

    Parameters
    ----------
    latent_data : 2D numpy array
        This is assumed to be the latent space for reference or input of 2D
    span : float or int
        The buffer distance to be padded along the min-max of latent_data

    """
    grid = np.mgrid[latent_data[:,0].min()-span:latent_data[:,0].max()+span:100j,
           latent_data[:,1].min()-span:latent_data[:,1].max()+span:100j]
    grid_2d = grid.reshape(2, -1).T
    return grid, grid_2d

def plot_contour(contour_data, grid, figsize=(16,9),cmap='Greys', colorbar=True, fig=None, ax=None):
    """
    Plot contour map.
    Grid can be easily generated via the `get_grid2d_latent`
    """
    levels = np.linspace(contour_data.min()*10,contour_data.max()*10,25)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    contour = ax.contourf(grid[0], grid[1], contour_data.reshape(100, 100)*10, levels=levels, cmap=cmap)
    if colorbar:
        fig.colorbar(contour)
    return fig, ax
