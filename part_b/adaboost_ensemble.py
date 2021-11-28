
# TODO: complete this file.

from sklearn.impute import KNNImputer
import torch
from torch import optim
from torch.autograd import Variable
import part_a.neural_network as nn
import part_a.item_response as ir
import part_a.knn as knn
from utils import *
import numpy as np

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
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0

    # Change to Float Tensor for PyTorch.
    # zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    # train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data,

def weighted_bagging(zero_train_matrix, weight_matrix, N):

    current_training_set = zero_train_matrix.copy()
    rng = np.random.default_rng()  # get a default random generator

    for i in range(N):
        n = rng.uniform(0, 1) #get a number between 1 and zero
        current_bottom = 0

        for sample_index in range(len(weight_matrix)):
            current_top = current_bottom + weight_matrix[sample_index]
            if current_bottom <= n <= current_top:
                current_training_set[i] = zero_train_matrix[sample_index]
                break
            current_bottom = current_top

    return  current_training_set


def update_model_weight(model,)

def evaluate_adaboost_ensemble():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    N = len(zero_train_matrix)
    initial_weight = 1 / N
    sample_weight_matrix = np.full((N, 1), initial_weight)

    model_weights = np.ones((3,1))

    for model_index in range(3):
        # bootstrap
        current_training_set = weighted_bagging(zero_train_matrix, sample_weight_matrix, N)

        current_training_set = torch.FloatTensor(current_training_set)
        train_matrix = torch.FloatTensor(train_matrix)
        # train
        if model_index == 0:
        # train knn
        elif model_index == 1:
        # train IRT
        elif model_index == 2:
        # train neural net
            model = nn.AutoEncoder(num_question=len(train_matrix[0]), k=10, p=0)
            print("training neural net")
            print("k* = " + str(10) + "; learning rate = " + str(0.05)
                  + "; num_epoch = " + str(20) + "; lamb=" + str(0.00025) + "; p=" + str(0))
            nn.train(model,
                     lr=0.05,
                     lamb=0.00025,
                     train_data=train_matrix,
                     zero_train_data=current_training_set,
                     valid_data=valid_data,
                     num_epoch=20)
            test_acc = nn.evaluate(model, current_training_set, test_data)
            print("Test Acc: {}".format(test_acc))

        # update model weight
        model_weights[model_index] = update_model_weight()





    # eva
    # update model weight
    # update sample weight

    knn_base_dist = np.random.randint(0, len(train_matrix))
    nn_base_dist = np.random.randint(0, len(train_matrix))
    # TODO: Implement IRT

    # Train
    nn_model = train_neural_network(zero_train_matrix[nn_base_dist], train_matrix)
    # TODO: Implement IRT

    # Predict
    knn_correct = 0
    nn_correct = 0
    correct = 0
    total = 0
    for i, u in enumerate(valid_data["user_id"]):
        q_id = valid_data["question_id"][i]

        knn_pred = predict_knn(train_matrix[u], q_id, train_matrix)
        nn_pred = predict_nn(zero_train_matrix[u], q_id, nn_model)
        # TODO: Implement IRT

        avg = (knn_pred + nn_pred) / 2
        # TODO: Implement IRT

        actual = bool(valid_data["is_correct"][i])
        if (avg >= 0.5) == actual:
            correct += 1
            total += 1
        if (knn_pred >= 0.5) == actual:
            knn_correct += 1
        if (nn_pred >= 0.5) == actual:
            nn_correct += 1
        print("Iteration: {}, Current correct: {}, Current kNN correct: {}, Current NN correct: {}".format(i, correct,
                                                                                                           knn_correct,
                                                                                                           nn_correct))
    return {
        "ensemble_acc": correct / float(total),
        "knn_bagged_acc": knn_correct / float(total),
        "nn_bagged_acc": nn_correct / float(total)
    }


def main():
    print(evaluate_adaboost_ensemble())


if __name__ == "__main__":
    main()