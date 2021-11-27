from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    copy_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0

    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return copy_train_matrix,zero_train_matrix, train_matrix, valid_data, test_data,

def main():
    copy_train_matrix, zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    #                                                            #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    #assign weight
    N = len(zero_train_matrix)
    initial_weight = 1/N
    sample_weight_matrix = np.full((N,1), initial_weight)

    # train

    # assign model weight
    correct_or_wrong = np.full((N,1), True)

    weighted_train_acc = compute_weighted_train_acc(sample_weight_matrix, correct_or_wrong)

    model_weight = 0.5*np.log(weighted_train_acc/(1-weighted_train_acc))

    # update sample weight
    updated_sample_weight = update_sample_weight(model_weight, sample_weight_matrix, correct_or_wrong)

    train_matrix = weighted_bootstrapping(copy_train_matrix,updated_sample_weight)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

def update_sample_weight(model_weight, sample_weight_matrix, correct_or_wrong):

    updated_weight_sum = 0
    for item_index in range(len(correct_or_wrong)):

        if correct_or_wrong[item_index] is True:
            sample_weight_matrix[item_index] = sample_weight_matrix[item_index] * (np.e**model_weight)
        elif correct_or_wrong[item_index] is False:
            sample_weight_matrix[item_index] = sample_weight_matrix[item_index] * ((-1)*(np.e ** model_weight))

        updated_weight_sum += sample_weight_matrix[item_index]
    sample_weight_matrix = sample_weight_matrix/updated_weight_sum

    return sample_weight_matrix

def compute_weighted_train_acc(sample_weight_matrix, correct_or_wrong):

    total_wrong = 0
    for item_index in range(len(sample_weight_matrix)):
        if correct_or_wrong[item_index] is False:
            total_wrong += sample_weight_matrix[item_index]

    weighted_train_acc = 1-total_wrong
    return weighted_train_acc

def weighted_bootstrapping(copy_train_matrix,sample_weight):

    updated_train_matrix = copy_train_matrix.copy()
    for item_index in range(len(copy_train_matrix)):

    return updated_train_matrix


if __name__ == "__main__":
    main()