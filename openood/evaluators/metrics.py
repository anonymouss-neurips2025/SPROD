import numpy as np
from sklearn import metrics

def noninf(arr):
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return finite_vals
    max_val = 2 * np.max(finite_vals)
    min_val = 2 * np.min(finite_vals)
    
    arr = np.where(arr == np.inf, max_val, arr)
    arr = np.where(arr == -np.inf, min_val, arr)
    arr = arr/(np.abs(arr).max()+1e-5)
    
    return arr

def compute_all_metrics(conf, label, pred, ood_as_positive=False):
    np.set_printoptions(precision=3)
    recall = 0.95
    conf = noninf(conf)
    pred = noninf(pred)
    conf_inds = np.argwhere(~np.isnan(conf)).ravel()
    pred_inds = np.argwhere(~np.isnan(pred)).ravel()
    union_inds = set(conf_inds).intersection(set(pred_inds))

    non_nan_inds = np.array(list(union_inds))
    if len(non_nan_inds) == 0:
        results = [np.nan,np.nan,np.nan,np.nan,np.nan]
        print("*"*40)
        print("Nan encountered!")
        print("*"*40)
    else:
        conf = conf[non_nan_inds]
        pred = pred[non_nan_inds]
        label = label[non_nan_inds]
        auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall, ood_as_positive)

        accuracy = acc(pred, label)

        results = [fpr, auroc, aupr_in, aupr_out, accuracy]

    return results


# accuracy
def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# fpr_recall
def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)
    gt[label == -1] = 0

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


# auc
def auc_and_fpr_recall(conf, label, tpr_th, ood_as_positive=False):

    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    if ood_as_positive:
        fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    else:
        fpr_list, tpr_list, thresholds = metrics.roc_curve(1 - ood_indicator, conf)

    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]


    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


# ccr_fpr
def ccr_fpr(conf, fpr, pred, label):
    ind_conf = conf[label != -1]
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    ood_conf = conf[label == -1]

    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    fp_num = int(np.ceil(fpr * num_ood))
    thresh = np.sort(ood_conf)[-fp_num]
    num_tp = np.sum((ind_conf > thresh) * (ind_pred == ind_label))
    ccr = num_tp / num_ind

    return ccr


def detection(ind_confidences,
              ood_confidences,
              n_iter=100000,
              return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta
