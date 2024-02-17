import torch
import numpy as np
import scipy


def get_dice_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    output = output.view(-1)
    mask = mask.view(-1)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice


def get_soft_dice(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average dice of B items
    """
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dice_item_thres = get_dice_threshold(output, mask, thres)
            dice_item_thres_list.append(dice_item_thres.data)
        dice_item_thres_mean = np.mean(dice_item_thres_list)
        dice_list.append(dice_item_thres_mean)

    return np.mean(dice_list)


def get_iou_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: iou of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)

    intersection = (output * mask).sum()
    total = (output + mask).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)

    return IoU


def get_soft_iou(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average iou of B items
    """
    iou_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        iou_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            iou_item_thres = get_iou_threshold(output, mask, thres)
            iou_item_thres_list.append(iou_item_thres)
        iou_item_thres_mean = np.mean(iou_item_thres_list)
        iou_list.append(iou_item_thres_mean)

    return np.mean(iou_list)

# =========== GED ============= #


def segmentation_scores(mask1, mask2):
    IoU = get_iou_threshold(mask1, mask2, threshold=0.5)
    return 1.0 - IoU


def generalized_energy_distance(label_list, pred_list):
    label_label_dist = [segmentation_scores(label_1, label_2) for i1, label_1 in enumerate(label_list)
                        for i2, label_2 in enumerate(label_list) if i1 != i2]
    pred_pred_dist = [segmentation_scores(pred_1, pred_2) for i1, pred_1 in enumerate(pred_list)
                      for i2, pred_2 in enumerate(pred_list) if i1 != i2]
    pred_label_list = [segmentation_scores(pred, label) for i, pred in enumerate(pred_list)
                       for j, label in enumerate(label_list)]
    GED = 2 * sum(pred_label_list) / len(pred_label_list) \
          - sum(label_label_dist) / len(label_label_dist) - sum(pred_pred_dist) / len(pred_pred_dist)
    return GED


def get_GED(batch_label_list, batch_pred_list):
    """
    :param batch_label_list: list_list
    :param batch_pred_list:
    :return:
    """
    batch_size = len(batch_pred_list)
    GED = 0.0
    for idx in range(batch_size):
        GED_temp = generalized_energy_distance(label_list=batch_label_list[idx], pred_list=batch_pred_list[idx])
        GED = GED + GED_temp
    return GED / batch_size


def get_uncertainty_metrics(predictions, labels, T):
    '''Calculates the uncertainty metrics
    Args:
        predictions: A numpy array of shape (N, C, H, W) or (N, T, C, H, W)
        labels: A numpy array of shape (N, H, W) used to calculate the Negative Log-Likelihood
        T: The number of initial heads to skip in the ensemble to calculate uncertainty
    Returns:
        A dictionary of metrics (Entropy, Mutual Information, Variance, Negative Log-Likelihood)
    '''
    # (N, num_heads, C, H, W)
    num_heads = predictions.shape[1]
    assert T < num_heads, 'SKIP_FIRST_T must be less than the number of heads'
    num_classes = predictions.shape[2]

    # these are uncertainty heatmaps
    entropy_maps = []
    variance_maps = []
    mi_maps = []
    # these are uncertainty metrics for each sample
    entropy_sum = []
    variance_sum = []
    mi_sum = []
    # area under layer agreement curve AULA
    aula_per_class = dict()
    for i in range(1, num_classes):  # ignore background
        aula_per_class[f'aula_{i}'] = []
    # calibration (NLL)
    nlls = []
        
    # convert labels to one hot
    # labels = np.eye(num_classes)[labels.astype(np.uint8)]  # (N, H, W) -> (N, H, W, C)
    # labels = np.transpose(labels, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)

    for predicted, label in zip(predictions, labels):
        # softmax along channel axis (NH, C, H, W)
        pred = scipy.special.softmax(predicted[T:, ...], axis=1)
        # average along layer ensemble heads. Keep only the last T heads
        # ([T:], C, H, W) -> (C, H, W)
        avg_pred = np.mean(pred, axis=0)

        # calculate entropy
        entropy = -np.sum(np.mean(pred, axis=0) * np.log(np.mean(pred, axis=0) + 1e-5), axis=0)
        entropy_maps.append(entropy)
        entropy_sum.append(np.sum(entropy))
        
        # calculate variance (after argmax on channel axis)
        variance = np.var(np.argmax(pred, axis=1), axis=0)
        variance_maps.append(variance)
        variance_sum.append(np.sum(variance))

        # calculate mutual information
        expected_entropy = -np.mean(np.sum(pred * np.log(pred + 1e-5), axis=1), axis=0)
        mi = entropy - expected_entropy
        mi_maps.append(mi)
        mi_sum.append(np.sum(mi))

        # calculate negative log-likelihood
        # label (C, H, W); avg_pred (C, H, W)
        nll = -np.mean(np.sum(label * np.log(avg_pred + 1e-5), axis=0))
        nlls.append(nll)
    
    metrics = {
        'entropy': entropy_sum,
        'variance': variance_sum,
        'mi': mi_sum,
        'nll': nlls
    }
    return metrics, entropy_maps, variance_maps, mi_maps