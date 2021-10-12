import numpy as np
from sklearn import metrics
import os.path as path
import pandas as pd
import os

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


def gss(y_true, y_pred, return_ss=False):
    """
    Gmean of sensitivity and specificity.
    If return_ss is True, returns the specificity and sensitivity
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    gss_ = np.sqrt(sens * spec)
    if return_ss:
        return gss_, sens, spec
    else:
        return gss_


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


# def calc_auroc(results_test, results_ood, exclude_keys=[]):
#     """
#     Given predictions on test and OOD, calculate the AUROC score.
#     Returns the AUROC,FPR80 and each metric name (epistemic, NLL, etc) used
#     """
#     # exclude mu and input
#     exclude_keys.append("mu")
#     exclude_keys.append("input")
#
#     test_scores, metric_names = calc_mean_results(
#         results_test, exclude_keys=exclude_keys
#     )
#     ood_scores, metric_names = calc_mean_results(results_ood, exclude_keys=exclude_keys)
#
#     fpr80_list = []
#     auroc_list = []
#
#     for metric_num, metric_name in enumerate(metric_names):
#         test_score = test_scores[metric_num]
#         ood_score = ood_scores[metric_num]
#
#         total_scores = np.concatenate((test_score, ood_score))
#         y_true = np.concatenate((np.zeros_like(test_score), np.ones_like(ood_score)))
#         fpr, tpr, thresholds = metrics.roc_curve(y_true, total_scores, pos_label=1)
#         auroc_score = metrics.roc_auc_score(y_true, total_scores)
#         fpr80 = calc_fpr80(fpr, tpr)
#         fpr80_list.append(np.round(fpr80, 4))
#         auroc_list.append(np.round(auroc_score, 4))
#
#     return auroc_list, fpr80_list, metric_names

#
# def calc_auprc(results_test, results_ood, exclude_keys=[]):
#     # exclude mu and input
#     exclude_keys.append("mu")
#     exclude_keys.append("input")
#
#     test_scores, metric_names = calc_mean_results(
#         results_test, exclude_keys=exclude_keys
#     )
#     ood_scores, metric_names = calc_mean_results(results_ood, exclude_keys=exclude_keys)
#
#     auprc_list = []
#     for metric_num, metric_name in enumerate(metric_names):
#         test_score = test_scores[metric_num]
#         ood_score = ood_scores[metric_num]
#
#         total_scores = np.concatenate((test_score, ood_score))
#         y_true = np.concatenate((np.zeros_like(test_score), np.ones_like(ood_score)))
#         precision, recall, thresholds = metrics.precision_recall_curve(
#             y_true, total_scores, pos_label=1
#         )
#         auprc_score = metrics.auc(recall, precision)
#         auprc_list.append(np.round(auprc_score, 4))
#
#     return auprc_list, metric_names


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


def calc_auroc(inliers, outliers, return_threshold=False):
    y_true = np.concatenate(
        (
            np.zeros(inliers.shape[0]),
            np.ones(outliers.shape[0]),
        )
    )
    y_scores = np.concatenate((inliers, outliers))

    # check if needed to return raw tpr-fpr thresholds
    if return_threshold:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        auroc = auc(fpr, tpr)

        return auroc, (fpr, tpr, thresholds)

    # return results
    else:
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


def calc_auprc(inliers, outliers):
    y_true = np.concatenate(
        (
            np.zeros(inliers.shape[0]),
            np.ones(outliers.shape[0]),
        )
    )
    y_scores = np.concatenate((inliers, outliers))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def calc_avgprc_perf(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)
    auroc_score = roc_auc_score(y_true, y_score)
    avgprc = average_precision_score(y_true, y_score)
    baseline = np.mean(y_true)

    return {
        "auroc": auroc_score,
        "auprc": auprc,
        "avgprc": avgprc,
        "baseline": baseline,
    }


def concat_ood_score(inliers, outliers, p_threshold=-1):
    y_true = np.concatenate(
        (
            np.zeros(inliers.shape[0]),
            np.ones(outliers.shape[0]),
        )
    )

    y_scores = np.concatenate((inliers, outliers))

    if p_threshold > 0:
        y_hard_pred = convert_hard_pred(y_scores, p_threshold=p_threshold)
        return y_true, y_scores, y_hard_pred
    else:
        return y_true, y_scores


def convert_hard_pred(prob, p_threshold=0.5):
    hard_inliers_test = np.piecewise(
        prob, [prob < p_threshold, prob >= p_threshold], [0, 1]
    ).astype(int)
    return hard_inliers_test


def get_indices_error(y_true, y_hard_pred, y_unc):

    indices_tp = np.argwhere((y_true == 1) & (y_hard_pred == 1))[:, 0]
    indices_tn = np.argwhere((y_true == 0) & (y_hard_pred == 0))[:, 0]
    indices_fp = np.argwhere((y_true == 0) & (y_hard_pred == 1))[:, 0]
    indices_fn = np.argwhere((y_true == 1) & (y_hard_pred == 0))[:, 0]
    indices_0_error = np.concatenate((indices_tp, indices_tn))
    indices_all_error = np.concatenate((indices_fp, indices_fn))

    error_type1 = np.concatenate(
        (np.ones(len(indices_fp)), np.zeros(len(indices_tp)))
    ).astype(int)
    error_type2 = np.concatenate(
        (np.ones(len(indices_fn)), np.zeros(len(indices_tn)))
    ).astype(int)
    error_all = np.abs((y_true - y_hard_pred))

    y_unc_type1 = np.concatenate((y_unc[indices_fp], y_unc[indices_tp]))
    y_unc_type2 = np.concatenate((y_unc[indices_fn], y_unc[indices_tn]))
    y_unc_all = y_unc.copy()

    return (
        indices_tp,
        indices_tn,
        indices_fp,
        indices_fn,
        indices_0_error,
        indices_all_error,
        error_type1,
        error_type2,
        error_all,
        y_unc_type1,
        y_unc_type2,
        y_unc_all,
    )


def retained_top_unc_indices(
    all_y_true, all_unc, retained_perc=0.6, return_referred=False
):
    """
    Gets a percentage of the retained indices sorted by ascending uncertainty.
    I.e the remaining data points gets referred away.

    Has the option to return both the retained and referred indices. By default returns only the retained indices.
    """
    # get size of retained index
    retained_index_size = int(np.round(len(all_y_true) * retained_perc))

    # sort by uncertainty in ascending (or if reverse, in descending) order
    retained_unc_indices = np.argsort(all_unc)[:retained_index_size]

    # get predictions of retained subsets
    retained_id_indices = np.argwhere(all_y_true[retained_unc_indices] == 0)[:, 0]
    retained_ood_indices = np.argwhere(all_y_true[retained_unc_indices] == 1)[:, 0]

    # handle returning retained and referred indices
    if not return_referred:
        return (retained_unc_indices, retained_id_indices, retained_ood_indices)

    else:
        referred_unc_indices = np.argsort(all_unc)[retained_index_size:]
        referred_id_indices = np.argwhere(all_y_true[referred_unc_indices] == 0)[:, 0]
        referred_ood_indices = np.argwhere(all_y_true[referred_unc_indices] == 1)[:, 0]

        return (retained_unc_indices, retained_id_indices, retained_ood_indices), (
            referred_unc_indices,
            referred_id_indices,
            referred_ood_indices,
        )


def retained_random_indices(all_y_true, retained_perc=0.6, return_referred=False):
    # get size of retained index
    retained_index_size = int(np.round(len(all_y_true) * retained_perc))

    # handle random referral
    retained_unc_indices = np.random.choice(
        np.arange(len(all_y_true)), size=retained_index_size, replace=False
    )

    # get predictions of retained subsets
    retained_id_indices = np.argwhere(all_y_true[retained_unc_indices] == 0)[:, 0]
    retained_ood_indices = np.argwhere(all_y_true[retained_unc_indices] == 1)[:, 0]

    if not return_referred:
        return (retained_unc_indices, retained_id_indices, retained_ood_indices)

    else:
        referred_unc_indices = np.array(
            [
                number
                for number in np.arange(len(all_y_true))
                if number not in retained_unc_indices
            ]
        )
        referred_id_indices = np.argwhere(all_y_true[referred_unc_indices] == 0)[:, 0]
        referred_ood_indices = np.argwhere(all_y_true[referred_unc_indices] == 1)[:, 0]

        return (retained_unc_indices, retained_id_indices, retained_ood_indices), (
            referred_unc_indices,
            referred_id_indices,
            referred_ood_indices,
        )


def evaluate_retained_unc(
    all_outprob_mean,
    all_hard_pred,
    all_y_true,
    all_unc=None,
    retained_percs=[0.6, 0.7, 0.8, 0.9, 1.0],
    keep_traces=True,
    *args,
    **kwargs,
):
    """
    Evaluate the classifier's performance along the scheme of retained/referral of uncertain predictions.
    Predictions with low uncertainties are retained while high uncertainties are `referred`.
    Classification performance is only evaluated on predictions with low uncertainties.
    Loops through the retained_perc and save results into a running list

    Returns a dictionary of classifier performances as retained perc. is varied.
    Has the option of returning traces (to plot ROC Curve and histograms) if required.
    """

    # prepare the loop
    auroc_retained_list = []
    f1_score_retained_list = []
    gss_score_retained_list = []
    sens_score_retained_list = []
    spec_score_retained_list = []
    auprc_retained_list = []
    avgprc_retained_list = []
    valid_retained_percs = []
    auroc_curve_list = []
    auroc_traces_list = []
    baselines_list = []

    # loop across retained percentages
    for retained_perc in retained_percs:
        try:
            (
                filtered_unc_indices,
                retained_id_indices,
                retained_ood_indices,
            ) = retained_top_unc_indices(
                all_y_true, all_unc, retained_perc=retained_perc, return_referred=False
            )

            retained_id_outprob_mean = all_outprob_mean[filtered_unc_indices][
                retained_id_indices
            ]
            retained_ood_outprob_mean = all_outprob_mean[filtered_unc_indices][
                retained_ood_indices
            ]

            retained_y_true = all_y_true[filtered_unc_indices][
                np.concatenate((retained_id_indices, retained_ood_indices))
            ].astype(int)
            retained_hard_pred = all_hard_pred[filtered_unc_indices][
                np.concatenate((retained_id_indices, retained_ood_indices))
            ].astype(int)

            # check for validity
            if retained_y_true.sum() >= 3:
                # calculate performance of retained pred.
                auroc_retained, auroc_curve = calc_auroc(
                    retained_id_outprob_mean,
                    retained_ood_outprob_mean,
                    return_threshold=True,
                )
                auprc_retained = calc_auprc(
                    retained_id_outprob_mean, retained_ood_outprob_mean
                )
                avgprc_retained = calc_avgprc(
                    retained_id_outprob_mean, retained_ood_outprob_mean
                )
                baseline_retained = np.mean(retained_y_true)
                f1_score_retained = f1_score(retained_y_true, retained_hard_pred)
                gss_score_retained, sens_retained, spec_retained = gss(
                    retained_y_true, retained_hard_pred, return_ss=True
                )

                # append to running list
                auroc_retained_list.append(auroc_retained)
                auprc_retained_list.append(auprc_retained)
                avgprc_retained_list.append(avgprc_retained)
                f1_score_retained_list.append(f1_score_retained)
                gss_score_retained_list.append(gss_score_retained)
                sens_score_retained_list.append(sens_retained)
                spec_score_retained_list.append(spec_retained)
                valid_retained_percs.append(retained_perc)
                auroc_curve_list.append(auroc_curve)
                auroc_traces_list.append(
                    [retained_id_outprob_mean, retained_ood_outprob_mean]
                )
                baselines_list.append(baseline_retained)
        except Exception as e:
            print(e)

    # save results
    results = {
        "auroc": auroc_retained_list,
        "auprc": auprc_retained_list,
        "avgprc": avgprc_retained_list,
        "f1_score": f1_score_retained_list,
        "gss": gss_score_retained_list,
        "sens": sens_score_retained_list,
        "spec": spec_score_retained_list,
        "valid_perc": valid_retained_percs,
        "baseline": baselines_list,
    }

    if keep_traces:
        results.update(
            {
                "auroc_curve": auroc_curve_list,
                "auroc_traces": auroc_traces_list,
            }
        )

    return results


def evaluate_retained_unc_v2(
    all_outprob_mean,
    all_hard_pred,
    all_y_true,
    all_unc=None,
    keep_traces=True,
    round_deci=0,
    *args,
    **kwargs,
):
    """
    Evaluate the classifier's performance along the scheme of retained/referral of uncertain predictions.
    Predictions with low uncertainties are retained while high uncertainties are `referred`.
    Classification performance is only evaluated on predictions with low uncertainties.
    Loops through the retained_perc and save results into a running list

    Returns a dictionary of classifier performances as retained perc. is varied.
    Has the option of returning traces (to plot ROC Curve and histograms) if required.
    """

    # prepare the loop
    auroc_retained_list = []
    f1_score_retained_list = []
    gss_score_retained_list = []
    sens_score_retained_list = []
    spec_score_retained_list = []
    auprc_retained_list = []
    avgprc_retained_list = []
    valid_retained_percs = []
    auroc_curve_list = []
    auroc_traces_list = []
    baselines_list = []
    threshold_list = []

    # obtain unc. thresholds
    thresholds = np.unique(all_unc)
    if round_deci > 0 and (len(thresholds) > round_deci):
        # thresholds = np.unique(all_unc.round(round_deci))
        thresholds = np.linspace(np.min(all_unc), np.max(all_unc), round_deci)

    # ensure max unc. is evaluated as a threshold
    # if np.max(all_unc) not in thresholds:
    #     thresholds = np.concatenate((thresholds, np.array([np.max(all_unc)])))

    # loop across thresholds
    for threshold in thresholds:
        try:
            filtered_unc_indices = np.argwhere(all_unc <= threshold)[:, 0]
            retained_y_trues = all_y_true[filtered_unc_indices]
            retained_id_indices = np.argwhere(retained_y_trues == 0)[:, 0]
            retained_ood_indices = np.argwhere(retained_y_trues == 1)[:, 0]

            retained_id_outprob_mean = all_outprob_mean[filtered_unc_indices][
                retained_id_indices
            ]
            retained_ood_outprob_mean = all_outprob_mean[filtered_unc_indices][
                retained_ood_indices
            ]

            retained_y_true = all_y_true[filtered_unc_indices][
                np.concatenate((retained_id_indices, retained_ood_indices))
            ].astype(int)
            retained_hard_pred = all_hard_pred[filtered_unc_indices][
                np.concatenate((retained_id_indices, retained_ood_indices))
            ].astype(int)
            retained_perc = len(filtered_unc_indices) / len(all_unc)

            # check for validity
            if retained_y_true.sum() >= 3:
                # calculate performance of retained pred.
                auroc_retained, auroc_curve = calc_auroc(
                    retained_id_outprob_mean,
                    retained_ood_outprob_mean,
                    return_threshold=True,
                )
                auprc_retained = calc_auprc(
                    retained_id_outprob_mean, retained_ood_outprob_mean
                )
                avgprc_retained = calc_avgprc(
                    retained_id_outprob_mean, retained_ood_outprob_mean
                )
                f1_score_retained = f1_score(retained_y_true, retained_hard_pred)
                gss_score_retained, sens_retained, spec_retained = gss(
                    retained_y_true, retained_hard_pred, return_ss=True
                )
                baseline_retained = np.mean(retained_y_true)

                # append to running list
                auroc_retained_list.append(auroc_retained)
                auprc_retained_list.append(auprc_retained)
                avgprc_retained_list.append(avgprc_retained)
                f1_score_retained_list.append(f1_score_retained)
                gss_score_retained_list.append(gss_score_retained)
                sens_score_retained_list.append(sens_retained)
                spec_score_retained_list.append(spec_retained)
                valid_retained_percs.append(retained_perc)
                auroc_curve_list.append(auroc_curve)
                auroc_traces_list.append(
                    [retained_id_outprob_mean, retained_ood_outprob_mean]
                )
                baselines_list.append(baseline_retained)
                threshold_list.append(threshold)
        except Exception as e:
            print(e)

    # save results
    results = {
        "auroc": auroc_retained_list,
        "auprc": auprc_retained_list,
        "avgprc": avgprc_retained_list,
        "f1_score": f1_score_retained_list,
        "gss": gss_score_retained_list,
        "sens": sens_score_retained_list,
        "spec": spec_score_retained_list,
        "valid_perc": valid_retained_percs,
        "baseline": baselines_list,
        "threshold": threshold_list,
    }

    if keep_traces:
        results.update(
            {
                "auroc_curve": auroc_curve_list,
                "auroc_traces": auroc_traces_list,
            }
        )

    return results


def evaluate_random_retained_unc(
    all_outprob_mean,
    all_hard_pred,
    all_y_true,
    retained_percs=[0.6, 0.7, 0.8, 0.9, 1.0],
    repetition=10,
):
    """
    Evaluate using random referral.
    Number of repetition of randomness can be adjusted.
    """

    # prepare the loop
    auroc_retained_list = []
    f1_score_retained_list = []
    gss_score_retained_list = []
    sens_score_retained_list = []
    spec_score_retained_list = []
    auprc_retained_list = []
    avgprc_retained_list = []
    valid_retained_percs = []
    baselines_list = []

    # loop across retained percentages
    for retained_perc in retained_percs:
        auroc_retained_list_rep = []
        f1_score_retained_list_rep = []
        gss_score_retained_list_rep = []
        sens_score_retained_list_rep = []
        spec_score_retained_list_rep = []
        auprc_retained_list_rep = []
        avgprc_retained_list_rep = []
        baselines_retained_list_rep = []

        for i in range(repetition):
            try:
                # get size of retained index
                retained_index_size = int(np.round(len(all_y_true) * retained_perc))

                # handle random referral
                filtered_unc_indices = np.random.choice(
                    np.arange(len(all_y_true)), size=retained_index_size, replace=False
                )

                # get predictions of retained subsets
                retained_id_indices = np.argwhere(
                    all_y_true[filtered_unc_indices] == 0
                )[:, 0]
                retained_ood_indices = np.argwhere(
                    all_y_true[filtered_unc_indices] == 1
                )[:, 0]

                retained_id_outprob_mean = all_outprob_mean[filtered_unc_indices][
                    retained_id_indices
                ]
                retained_ood_outprob_mean = all_outprob_mean[filtered_unc_indices][
                    retained_ood_indices
                ]

                retained_y_true = all_y_true[filtered_unc_indices][
                    np.concatenate((retained_id_indices, retained_ood_indices))
                ].astype(int)
                retained_hard_pred = all_hard_pred[filtered_unc_indices][
                    np.concatenate((retained_id_indices, retained_ood_indices))
                ].astype(int)

                # check for validity
                if retained_y_true.sum() >= 3:
                    # calculate performance of retained pred.
                    auroc_retained = calc_auroc(
                        retained_id_outprob_mean,
                        retained_ood_outprob_mean,
                        return_threshold=False,
                    )
                    auprc_retained = calc_auprc(
                        retained_id_outprob_mean, retained_ood_outprob_mean
                    )
                    avgprc_retained = calc_avgprc(
                        retained_id_outprob_mean, retained_ood_outprob_mean
                    )
                    f1_score_retained = f1_score(retained_y_true, retained_hard_pred)
                    gss_score_retained, sens_retained, spec_retained = gss(
                        retained_y_true, retained_hard_pred, return_ss=True
                    )
                    baseline_retained = np.mean(retained_y_true)

                    # append to running list
                    auroc_retained_list_rep.append(auroc_retained)
                    auprc_retained_list_rep.append(auprc_retained)
                    avgprc_retained_list_rep.append(avgprc_retained)
                    f1_score_retained_list_rep.append(f1_score_retained)
                    gss_score_retained_list_rep.append(gss_score_retained)
                    sens_score_retained_list_rep.append(sens_retained)
                    spec_score_retained_list_rep.append(spec_retained)
                    baselines_retained_list_rep.append(baseline_retained)

            except Exception as e:
                print(e)

        # append to running list
        auroc_retained_list.append(np.mean(auroc_retained_list_rep, axis=0))
        auprc_retained_list.append(np.mean(auprc_retained_list_rep, axis=0))
        avgprc_retained_list.append(np.mean(avgprc_retained_list_rep, axis=0))
        f1_score_retained_list.append(np.mean(f1_score_retained_list_rep, axis=0))
        gss_score_retained_list.append(np.mean(gss_score_retained_list_rep, axis=0))
        sens_score_retained_list.append(np.mean(sens_score_retained_list_rep, axis=0))
        spec_score_retained_list.append(np.mean(spec_score_retained_list_rep, axis=0))
        baselines_list.append(np.mean(baselines_retained_list_rep, axis=0))
        valid_retained_percs.append(retained_perc)

    # save results
    results = {
        "auroc": auroc_retained_list,
        "auprc": auprc_retained_list,
        "avgprc": avgprc_retained_list,
        "f1_score": f1_score_retained_list,
        "gss": gss_score_retained_list,
        "sens": sens_score_retained_list,
        "spec": spec_score_retained_list,
        "valid_perc": valid_retained_percs,
        "baseline": baselines_list,
    }

    return results


# AUPR PERF
def evaluate_misclas_detection(y_true, y_hard_pred, y_unc, return_boxplot=True):
    """
    Evaluates misclassification detection results using AUROC/AUPRC/AVGPRC.
    Further divides into various types of error : type1,type2 and all combined.
    If needed, also returns the raw unc. score for each type of error to be plotted in box plot.
    """
    # split into indices
    (
        indices_tp,
        indices_tn,
        indices_fp,
        indices_fn,
        indices_0_error,
        indices_all_error,
        error_type1,
        error_type2,
        error_all,
        y_unc_type1,
        y_unc_type2,
        y_unc_all,
    ) = get_indices_error(y_true, y_hard_pred, y_unc)
    final_results = {}
    # get avg prc performance
    try:
        type1_err_perf = calc_avgprc_perf(error_type1, y_unc_type1)
        type2_err_perf = calc_avgprc_perf(error_type2, y_unc_type2)
        all_err_perf = calc_avgprc_perf(error_all, y_unc_all)
        final_results = {
            "type1": type1_err_perf,
            "type2": type2_err_perf,
            "all_err": all_err_perf,
        }

        # y_unc for box plot
        if return_boxplot:
            y_unc_boxplot = {
                "type1": [y_unc[indices_tp], y_unc[indices_fp]],
                "type2": [y_unc[indices_tn], y_unc[indices_fn]],
                "all_err": [y_unc[indices_0_error], y_unc[indices_all_error]],
            }
            final_results.update({"y_unc_boxplot": y_unc_boxplot})

    except Exception as e:
        print(e)
        try:
            all_err_perf = calc_avgprc_perf(error_all, y_unc_all)
            final_results = {
                "all_err": all_err_perf,
            }
            if return_boxplot:
                y_unc_boxplot = {
                    "type1": [y_unc[indices_tp], y_unc[indices_fp]],
                    "type2": [y_unc[indices_tn], y_unc[indices_fn]],
                    "all_err": [y_unc[indices_0_error], y_unc[indices_all_error]],
                }
                final_results.update({"y_unc_boxplot": y_unc_boxplot})
        except Exception as e:
            print(e)
            all_err_perf = {
                "auroc": np.nan,
                "auprc": np.nan,
                "avgprc": np.nan,
                "baseline": np.nan,
            }
            final_results = {
                "all_err": all_err_perf,
            }
            if return_boxplot:
                y_unc_boxplot = {
                    "type1": [np.array([]), np.array([])],
                    "type2": [np.array([]), np.array([])],
                    "all_err": [np.array([]), np.array([])],
                }
                final_results.update({"y_unc_boxplot": y_unc_boxplot})

    return final_results


def summarise_retained_perf(retained_res, flatten_key=False):
    """
    Summarise the performance based on the retained/referrral scheme.
    """
    perf_keys = ["auroc", "auprc", "avgprc", "f1_score", "gss"]
    weighted_perf = {
        key: np.average(
            retained_res[key],
            weights=retained_res["valid_perc"],
        )
        for key in perf_keys
    }
    max_perf = {
        key: np.max(
            retained_res[key],
        )
        for key in perf_keys
    }
    baseline_perf = {key: retained_res[key][-1] for key in perf_keys}

    # flatten key of weighted/max/baseline so that only one level of dict results exist
    if not flatten_key:
        return {
            "weighted": weighted_perf,
            "max": max_perf,
            "baseline": baseline_perf,
        }
    else:
        flatten_res = {}
        flatten_res.update({"weighted-" + key: weighted_perf[key] for key in perf_keys})
        flatten_res.update({"max-" + key: max_perf[key] for key in perf_keys})
        flatten_res.update({"baseline-" + key: baseline_perf[key] for key in perf_keys})

        return flatten_res
