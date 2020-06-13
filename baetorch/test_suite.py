import numpy as np
from .plotting import plot_samples_img, plot_roc_curve, plot_prc_curve, plot_output_distribution, plot_calibration_curve, plot_train_loss, plot_latent
from .evaluation import calc_auroc, calc_auprc, save_csv_metrics, save_csv_distribution, calc_variance_dataset

def remove_nan(predict_res):
    predict_res['bce_mean'] = np.nan_to_num(predict_res['bce_mean'])
    predict_res['bce_var'] = np.nan_to_num(predict_res['bce_var'])
    predict_res['bce_waic'] = np.nan_to_num(predict_res['bce_waic'])
    return predict_res

def run_test_model(bae_models:list, id_data_test, ood_data_list:list=[], id_data_name:str="TEST", ood_data_names:list=[], output_reshape_size:tuple=(),
                   dist_exclude_keys:list = [],
                   exclude_keys:list =[], save_csv = False
                   ):
    #check if ood dataset is provided
    if len(ood_data_list) == 0:
        ood_provided = False
    else:
        ood_provided = True
        if len(ood_data_names) == 0: #resort to default names for OOD
            if len(ood_data_list) > 1:
                ood_data_names = ["OOD_"+str(i) for i in range(len(ood_data_list))]
            else:
                ood_data_names = ["OOD"]

    #run outer loop for each bae model, inner loop for each ood dataset
    for bae_model in bae_models:
        #compute model outputs
        predict_test = bae_model.predict(id_data_test)
        predict_test = np.nan_to_num(predict_test)
        predict_test = remove_nan(predict_test)

        predict_ood_list = [bae_model.predict(ood_data) for ood_data in ood_data_list]
        predict_ood_list = [remove_nan(predict_ood) for predict_ood in predict_ood_list]

        #plot reconstruction image of test set
        plot_samples_img(data=predict_test, reshape_size=output_reshape_size, savefile=bae_model.model_name + "-" + "TEST" + "-samples" + ".png")


        #evaluate performance curves by comparing against OOD datasets
        if ood_provided:
            for predict_ood,ood_data_name in zip(predict_ood_list,ood_data_names):


                plot_roc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUROC"+".png")
                plot_prc_curve(predict_test,predict_ood, title=bae_model.model_name +"-"+ood_data_name, savefile=bae_model.model_name +"-"+ood_data_name+"-AUPRC"+".png")

                #evaluation
                auroc_list, fpr80_list, metric_names= calc_auroc(predict_test,predict_ood, exclude_keys=exclude_keys)
                auprc_list, metric_names= calc_auprc(predict_test,predict_ood, exclude_keys=exclude_keys)
                plot_samples_img(data=predict_ood, reshape_size=output_reshape_size, savefile=bae_model.model_name + "-" + ood_data_name + "-samples" + ".png")
                #save performance results as csv
                save_csv_metrics(id_data_name+"_"+ood_data_name,bae_model.model_name , auroc_list, auprc_list, fpr80_list, metric_names)

            #plot and compare distributions of data per model
            plot_output_distribution(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names,exclude_keys=dist_exclude_keys, savefile=bae_model.model_name +"-dist"+".png")
            output_means, output_medians, dist_metric_names = calc_variance_dataset(predict_test,*predict_ood_list,legends=["TEST"]+ood_data_names, exclude_keys=["aleatoric_var","waic_se","nll_homo_mean","nll_homo_var","waic_homo","waic_sigma"])

            save_csv_distribution(id_data_name,bae_model.model_name, output_means, output_medians, dist_metric_names)

        if bae_model.homoscedestic_mode != "none":
            homoscedestic_noise = bae_model.get_homoscedestic_noise()
            print(bae_model.model_name +" Homo-noise:"+str(homoscedestic_noise))

        #plot miscellaneous
        plot_calibration_curve(bae_model, id_data_test)
        plot_train_loss(model=bae_model,savefile=bae_model.model_name+"-loss.png")
