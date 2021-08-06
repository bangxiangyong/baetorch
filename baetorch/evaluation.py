import numpy as np
from sklearn import metrics
import os.path as path
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, average_precision_score


def create_dir(folder="plots"):
    if os.path.exists(folder) == False:
        os.mkdir(folder)


def calc_mean_results(predict_results, exclude_keys=[]):

    keys = list(predict_results.keys())

    # filter exclude
    filtered_keys = [key for key in keys if key not in exclude_keys]
    filtered_key_ids = [
        key_id for key_id, key in enumerate(keys) if key in filtered_keys
    ]
    # filtered_values = np.array([values[key_id] for key_id in filtered_key_ids])
    filtered_values = {
        key: val for key, val in predict_results.items() if key in filtered_keys
    }
    # values = np.array([value.mean(1) for value in list(predict_results.values())])
    filtered_values = np.array(
        [value.mean(1) for value in list(filtered_values.values())]
    )

    return filtered_values, filtered_keys


def calc_variance_dataset(*args, legends=["TEST", "OOD"], exclude_keys=[], savefile=""):
    # exclude mu and input
    exclude_keys.append("mu")
    exclude_keys.append("input")

    num_datasets = len(args)

    means = []
    variance = []

    for dataset_id in range(num_datasets):
        metrics_mean, axis_titles = calc_mean_results(
            args[dataset_id], exclude_keys=exclude_keys
        )
        means.append(
            {"dataset": legends[dataset_id], "values": np.mean(metrics_mean, 1)}
        )
        variance.append(
            {"dataset": legends[dataset_id], "values": np.var(metrics_mean, 1)}
        )

    return means, variance, axis_titles


def calc_fpr80(fpr, tpr):
    # gets the false positive rate at 80% true positive rate
    tpr_80 = np.argwhere(tpr >= 0.8)
    if len(tpr_80) >= 1:
        return fpr[tpr_80[0]][0]
    else:
        return 1


def calc_auroc(results_test, results_ood, exclude_keys=[]):
    """
    Given predictions on test and OOD, calculate the AUROC score.
    Returns the AUROC,FPR80 and each metric name (epistemic, NLL, etc) used
    """
    # exclude mu and input
    exclude_keys.append("mu")
    exclude_keys.append("input")

    test_scores, metric_names = calc_mean_results(
        results_test, exclude_keys=exclude_keys
    )
    ood_scores, metric_names = calc_mean_results(results_ood, exclude_keys=exclude_keys)

    fpr80_list = []
    auroc_list = []

    for metric_num, metric_name in enumerate(metric_names):
        test_score = test_scores[metric_num]
        ood_score = ood_scores[metric_num]

        total_scores = np.concatenate((test_score, ood_score))
        y_true = np.concatenate((np.zeros_like(test_score), np.ones_like(ood_score)))
        fpr, tpr, thresholds = metrics.roc_curve(y_true, total_scores, pos_label=1)
        auroc_score = metrics.roc_auc_score(y_true, total_scores)
        fpr80 = calc_fpr80(fpr, tpr)
        fpr80_list.append(np.round(fpr80, 4))
        auroc_list.append(np.round(auroc_score, 4))

    return auroc_list, fpr80_list, metric_names


def calc_auprc(results_test, results_ood, exclude_keys=[]):
    # exclude mu and input
    exclude_keys.append("mu")
    exclude_keys.append("input")

    test_scores, metric_names = calc_mean_results(
        results_test, exclude_keys=exclude_keys
    )
    ood_scores, metric_names = calc_mean_results(results_ood, exclude_keys=exclude_keys)

    auprc_list = []
    for metric_num, metric_name in enumerate(metric_names):
        test_score = test_scores[metric_num]
        ood_score = ood_scores[metric_num]

        total_scores = np.concatenate((test_score, ood_score))
        y_true = np.concatenate((np.zeros_like(test_score), np.ones_like(ood_score)))
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_true, total_scores, pos_label=1
        )
        auprc_score = metrics.auc(recall, precision)
        auprc_list.append(np.round(auprc_score, 4))

    return auprc_list, metric_names


def get_homoscedestic_noise(bae):
    return bae.get_homoscedestic_noise()[0].mean()


def save_csv_metrics(
    train_set_name,
    model_name,
    auroc_metrics,
    auprc_metrics,
    fpr80_metrics,
    metric_names,
    savefolder="results",
):
    create_dir(savefolder)
    model_results = {"MODEL": model_name}
    auroc_dict = {
        metric_name + "_AUROC": auroc_metrics[metric_id]
        for metric_id, metric_name in enumerate(metric_names)
    }
    auprc_dict = {
        metric_name + "_AUPRC": auprc_metrics[metric_id]
        for metric_id, metric_name in enumerate(metric_names)
    }
    fpr80_dict = {
        metric_name + "_FPR80": fpr80_metrics[metric_id]
        for metric_id, metric_name in enumerate(metric_names)
    }
    model_results.update({"BEST_AUROC": np.max(auroc_metrics)})
    model_results.update({"BEST_AUPRC": np.max(auprc_metrics)})
    model_results.update({"BEST_FPR80": np.min(fpr80_metrics)})

    model_results.update(auroc_dict)
    model_results.update(auprc_dict)
    model_results.update(fpr80_dict)

    model_results_pd = pd.DataFrame.from_dict([model_results])

    # check for file exist
    save_path = savefolder + "/" + train_set_name + "_results.csv"
    csv_exists = path.exists(save_path)
    csv_mode = "a" if csv_exists else "w"
    header_mode = False if csv_exists else True
    model_results_pd.to_csv(save_path, mode=csv_mode, header=header_mode, index=False)

    return model_results_pd


def save_csv_distribution(
    train_set_name,
    model_name,
    mean_dict,
    variance_dict,
    metric_names,
    savefolder="results",
):
    create_dir(savefolder)
    model_results = {"MODEL": model_name}
    mean_dict = {
        mean_dict[dataset_id]["dataset"]
        + "_"
        + metric_name
        + "_MEAN": mean_dict[dataset_id]["values"][metric_id]
        for metric_id, metric_name in enumerate(metric_names)
        for dataset_id in range(len(mean_dict))
    }
    variance_dict = {
        variance_dict[dataset_id]["dataset"]
        + "_"
        + metric_name
        + "_VARIANCE": variance_dict[dataset_id]["values"][metric_id]
        for metric_id, metric_name in enumerate(metric_names)
        for dataset_id in range(len(variance_dict))
    }

    model_results.update(mean_dict)
    model_results.update(variance_dict)

    model_results_pd = pd.DataFrame.from_dict([model_results])

    # check for file exist
    save_path = savefolder + "/" + train_set_name + "_distribution.csv"
    csv_exists = path.exists(save_path)
    csv_mode = "a" if csv_exists else "w"
    header_mode = False if csv_exists else True
    model_results_pd.to_csv(save_path, mode=csv_mode, header=header_mode, index=False)

    return model_results_pd


def calc_auroc(inliers, outliers):
    y_true = np.concatenate(
        (
            np.zeros(inliers.shape[0]),
            np.ones(outliers.shape[0]),
        )
    )
    y_scores = np.concatenate((inliers, outliers))

    auroc = roc_auc_score(y_true, y_scores)
    return auroc


def calc_avgprc(inliers, outliers):
    y_true = np.concatenate(
        (
            np.zeros(inliers.shape[0]),
            np.ones(outliers.shape[0]),
        )
    )
    y_scores = np.concatenate((inliers, outliers))

    avgprc = average_precision_score(y_true, y_scores)
    return avgprc
