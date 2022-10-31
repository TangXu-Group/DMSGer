import numpy as np
import matplotlib.pyplot as plt
import torch
def performance(predict_labels, gt_labels, class_num):
    matrix = np.zeros((class_num, class_num))
    predict_labels = torch.max(predict_labels,dim=1)[1]
    
    for j in range(len(predict_labels)):
        o = predict_labels[j]
        q = gt_labels[j]
        if q == 0:
            continue
        matrix[o-1, q-1] += 1
#     plt.imshow(matrix)
#     plt.show()
    OA = np.sum(np.trace(matrix)) / np.sum(matrix)

    
    ac_list = np.zeros((class_num))
    for k in range(len(matrix)):
        ac_k = matrix[k, k] / sum(matrix[:, k])
        ac_list[k] = round(ac_k,4)
    
    AA = np.mean(ac_list)

    
    mm = 0
    for l in range(matrix.shape[0]):
        mm += np.sum(matrix[l]) * np.sum(matrix[:, l])
    pe = mm / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    
    
    return OA, AA, kappa, ac_list
