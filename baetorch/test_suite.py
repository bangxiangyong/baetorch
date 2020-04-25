import numpy as np
from bnn.develop.bayesian_autoencoders.plotting import *
from bnn.develop.bayesian_autoencoders.evaluation import *

def remove_nan(predict_res):
    predict_res['bce_mean'] = np.nan_to_num(predict_res['bce_mean'])
    predict_res['bce_var'] = np.nan_to_num(predict_res['bce_var'])
    predict_res['bce_waic'] = np.nan_to_num(predict_res['bce_waic'])
    return predict_res

def run_test_model(bae_models:list,id_data_test, ood_data_list:list, id_data_name:str="", ood_data_names:list=["OOD"],
                   dist_exclude_keys:list = ["aleatoric_var","waic_se","nll_homo_var","waic_homo","waic_sigma"],
                   exclude_keys:list =[]
                   ):
    for bae_model in bae_models:

        #compute model outputs
        predict_test = bae_model.predict(id_data_test)
        # predict_test = np.nan_to_num(predict_test,nan=0)
        predict_test = np.nan_to_num(predict_test)
        predict_test = remove_nan(predict_test)

        # predict_ood_list = [bae_model.predict(ood_data) for ood_data in ood_data_list]
        predict_ood_list = [bae_model.predict(ood_data) for ood_data in ood_data_list]
        predict_ood_list = [remove_nan(predict_ood) for predict_ood in predict_ood_list]

        #plot reconstruction image of test set
        plot_samples_img(data=predict_test, reshape_size=(28,28), savefile=bae_model.model_name +"-"+"TEST"+"-samples"+".png")

        #evaluate performance curves by comparing against OOD datasets
        for predict_ood,ood_data_name in zip(predict_ood_list,ood_data_names):

            plot_roc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUROC"+".png")
            plot_prc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUPRC"+".png")

            #evaluation
            auroc_list, fpr80_list, metric_names= calc_auroc(predict_test,predict_ood, exclude_keys=exclude_keys)
            auprc_list, metric_names= calc_auprc(predict_test,predict_ood, exclude_keys=exclude_keys)
            plot_samples_img(data=predict_ood, reshape_size=(28,28), savefile=bae_model.model_name +"-"+ood_data_name+"-samples"+".png")
            #save performance results as csv
            save_csv_metrics(id_data_name+"_"+ood_data_name,bae_model.model_name , auroc_list, auprc_list, fpr80_list, metric_names)

        #plot and compare distributions of data per model
        plot_output_distribution(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names,exclude_keys=dist_exclude_keys, savefile=bae_model.model_name +"-dist"+".png")
        output_means, output_medians, dist_metric_names = calc_variance_dataset(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names, exclude_keys=["aleatoric_var","waic_se","nll_homo_mean","nll_homo_var","waic_homo","waic_sigma"])
        save_csv_distribution(id_data_name,bae_model.model_name, output_means, output_medians, dist_metric_names)

        if bae_model.homoscedestic_mode != "none":
            homoscedestic_noise = bae_model.get_homoscedestic_noise()
            print(bae_model.model_name +" Homo-noise:"+str(homoscedestic_noise))

        plot_calibration_curve(bae_model, id_data_test)
        plot_train_loss(model=bae_model,savefile=bae_model.model_name+"-loss.png")
