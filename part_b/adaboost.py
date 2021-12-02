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
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0

    # Change to Float Tensor for PyTorch.
    # zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    # train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data, train_data


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def weighted_bagging(train_matrix, weight_matrix, N):
    bagging_order = weight_matrix.copy()
    current_training_set = train_matrix.copy()
    rng = np.random.default_rng()  # get a default random generator

    for i in range(N):
        n = rng.uniform(0, 1)  # get a number between 1 and zero
        current_bottom = 0

        for sample_index in range(len(weight_matrix)):
            current_top = current_bottom + weight_matrix[sample_index]
            if current_bottom <= n <= current_top:
                current_training_set[i] = train_matrix[sample_index]
                bagging_order[i] = sample_index
                break
            current_bottom = current_top

    current_zero_training_set = current_training_set.copy()
    current_zero_training_set[np.isnan(current_training_set)] = 0
    return current_training_set, current_zero_training_set, bagging_order


def update_sample_weight(sample_weight_matrix, wrong_samples, model_weight):
    updated_weight_sum = 0
    for item_index in range(len(wrong_samples)):

        if wrong_samples[item_index] > 0:  # samples that got wrong need more focus, increase weight
            u = wrong_samples[item_index]
            k = np.sqrt(u.item(0))
            for i in range(int(k)):
                sample_weight_matrix[item_index] += sample_weight_matrix[item_index] * (np.e ** model_weight)
        elif wrong_samples[item_index] <= 0:  # samples that got correct can predict easily, decrease weight
            sample_weight_matrix[item_index] = sample_weight_matrix[item_index] * (np.e ** -model_weight)

        updated_weight_sum += sample_weight_matrix[item_index]
    sample_weight_matrix = sample_weight_matrix / updated_weight_sum

    return sample_weight_matrix


def train_knn(current_training_set, current_train_data, valid_data, test_data):
    # train knn
    k_values = [48, 49, 50, 51, 52, 55]
    accuracies_user = []

    for i in k_values:
        accuracies_user.append(knn.knn_impute_by_user(current_training_set, valid_data, i))
    k_user = accuracies_user.index(max(accuracies_user))
    chosen_k = k_values[k_user]
    test_acc = knn.knn_impute_by_user(current_training_set, test_data, chosen_k)
    print("Highest k value for knn_impute_by_user: ", chosen_k)
    print("Test Accuracy for this value of k: ", test_acc)

    valid_acc, wrong, unused_wpu = evaluate_knn_by_user(current_training_set, valid_data, chosen_k)
    train_acc, train_wrong, train_wpu = evaluate_knn_by_user(current_training_set, current_train_data, chosen_k)
    return valid_acc, train_wrong, train_wpu


def evaluate_knn_by_user(train_data, valid_data, k):
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(train_data)

    total = 0
    total_array = np.zeros((542, 1))
    correct = 0
    correct_array = np.zeros((542, 1))
    wrong = np.zeros((542, 1))
    for i, q in enumerate(valid_data["question_id"]):  # i=index q=q_id
        u = valid_data["user_id"][i]
        prediction = mat[u][q]
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0

        if valid_data["is_correct"][i] == prediction:
            correct += 1
            correct_array[u] += 1
        else:
            wrong[u] += 1
        total += 1
        total_array[u] += 1

    wpu = correct_array / total_array

    acc = correct / float(total)
    print("Validation Accuracy: {}".format(acc))
    return acc, wrong, wpu


def train_irt(train_data, val_data, test_data):
    lr, iterations = 0.01, 20

    theta, beta, val_log_likelihood, train_log_likelihood = \
        ir.irt(train_data, val_data, lr, iterations)
    val_score, wrong, unused_wpu = evaluate_irt(data=val_data, theta=theta, beta=beta)
    test_score, test_wrong, unused_wpu = evaluate_irt(data=test_data, theta=theta, beta=beta)
    train_score, wrong, train_wpu = evaluate_irt(data=train_data, theta=theta, beta=beta)

    print("Validation Accuracy: ", val_score)
    print("Test Accuracy: ", test_score)

    return val_score, wrong, train_wpu, theta, beta, test_wrong


def evaluate_irt(data, theta, beta):
    total = 0
    correct = 0
    pred = []
    prediction = 0
    wrong_a = np.zeros((542, 1))
    correct_a = np.zeros((542, 1))
    total_a = np.zeros((542, 1))
    for i, q in enumerate(data["question_id"]):  # i=index q=q_id
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        if p_a < 0.5:
            prediction = 0
        elif p_a >= 0.5:
            prediction = 1
        if prediction == data["is_correct"][i]:
            correct += 1
            correct_a[u] += 1
        else:
            wrong_a[u] += 1

        total_a[u] += 1
        total += 1
    weight_per_user = correct_a / total_a

    return correct / float(total), wrong_a, weight_per_user


def train_neural_net(train_data, zero_train_data, current_train_data, valid_data, test_data):
    train_data = torch.FloatTensor(train_data)
    zero_train_data = torch.FloatTensor(zero_train_data)

    model = nn.AutoEncoder(num_question=len(train_data[0]), k=10)
    print("training neural net")
    print("k* = " + str(10) + "; learning rate = " + str(0.05)
          + "; num_epoch = " + str(20) + "; lamb=" + str(0.00025) + "; p=" + str(0))
    nn.train(model,
             lr=0.05,
             lamb=0.00025,
             train_data=train_data,
             zero_train_data=zero_train_data,
             valid_data=valid_data,
             num_epoch=5)
    test_acc, wrong, test_wpu = evaluate_nn(model, zero_train_data, test_data)
    print("Test Acc: {}".format(test_acc))

    valid_acc, wrong, valid_wpu = evaluate_nn(model, zero_train_data, valid_data)
    train_acc, train_wrong, train_wpu = evaluate_nn(model, zero_train_data, current_train_data)
    return model, valid_acc, train_wrong, train_wpu


def evaluate_nn(model, train_data, valid_data):
    model.eval()

    total = 0
    correct = 0
    total_array = np.zeros((542, 1))
    correct_array = np.zeros((542, 1))
    wrong = np.zeros((542, 1))

    for i, u in enumerate(valid_data["user_id"]):  # get current index on valid data's user_id line
        inputs = Variable(train_data[u]).unsqueeze(0)  # get that user's train data
        output = model(inputs)  # generate that user's prediction

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
            correct_array[u] += 1
        else:
            wrong[u] += 1

        total += 1
        total_array[u] += 1
    wpu = correct_array / total_array
    return correct / float(total), wrong, wpu


def sparse_martix_to_csv_data(matrix, train_array):
    index = 0
    for u in range(len(matrix)):
        for q in range(len(matrix[u])):
            if not (np.isnan(matrix[u][q])):
                train_array["user_id"][index] = u
                train_array["question_id"][index] = q
                train_array["is_correct"][index] = matrix[u][q]
                index += 1
            if index == len(train_array):
                break
        if index == len(train_array):
            break
    return train_array


def run_adaboost_ensemble():
    # set initial parameters, load data
    zero_train_matrix, train_matrix, valid_data, test_data, train_data = load_data()
    N = len(zero_train_matrix)
    initial_weight = 1 / N
    sample_weight_matrix = np.full((N, 1), initial_weight)

    model_weight = [1, 1, 1]
    model_weight_per_user = np.zeros((3, 542))
    models = []
    train_wrong = np.zeros((542, 1))
    train_wpu = np.zeros((542, 1))
    train_data_array = []
    for model_index in range(3):
        # bootstrap
        current_training_set, current_zero_training_set, bagging_order = \
            weighted_bagging(train_matrix, sample_weight_matrix, N)

        # current_training_set[0] = train_matrix[0]
        # current_zero_training_set[0] = zero_train_matrix[0]
        print(np.shape(current_training_set))
        combined_training_set = np.concatenate((train_matrix, current_training_set), axis=0)
        combined_zero_training_set = np.concatenate((zero_train_matrix, current_zero_training_set), axis=0)

        valid_acc = 0
        combined_train_data = train_data.copy()
        combined_train_data = sparse_martix_to_csv_data(combined_zero_training_set, combined_train_data)

        # train
        if model_index == 0:
            knn_train_set = combined_training_set.copy()
            train_data_array.append(combined_training_set)
            valid_acc, train_wrong, train_wpu = train_knn(current_training_set, combined_train_data, valid_data,
                                                          test_data)
            nbrs = KNNImputer(n_neighbors=26)
            # We use NaN-Euclidean distance measure.
            models.append(nbrs)
        elif model_index == 1:
            # train IRT
            valid_acc, train_wrong, train_wpu, theta, beta, test_wrong = \
                train_irt(combined_train_data, valid_data, test_data)
            irt_wrong = test_wrong
            models.append((theta, beta))
            train_data_array.append(combined_train_data)
        elif model_index == 2:
            # train neural net
            print(np.shape(current_training_set))
            model, valid_acc, train_wrong, train_wpu = train_neural_net(combined_training_set,
                                                                        combined_zero_training_set,
                                                                        combined_train_data,
                                                                        valid_data,
                                                                        test_data)
            models.append(model)
            train_data_array.append(combined_zero_training_set)

        model_weight[model_index] = 0.5 * np.log(valid_acc / (1 - valid_acc))

        # update sample weight
        if model_index != 2:
            sample_weight_matrix = update_sample_weight(sample_weight_matrix, train_wrong, model_weight[model_index])
        train_wpu = train_wpu.transpose()
        np.copyto(model_weight_per_user[model_index], train_wpu)
        if model_index == 2:
            print("chec")
            model_weight_per_user = model_weight_per_user.transpose()
    valid_acc = evaluate_adaboost_ensemble(models, model_weight_per_user, train_data_array, valid_data, irt_wrong)
    test_acc = evaluate_adaboost_ensemble(models, model_weight_per_user, train_data_array, test_data, irt_wrong)
    return str(test_acc)


def pick_suited_model(u, model_weight_per_user):
    model_weight_of_u = model_weight_per_user[u]
    max = 0
    max_index = 0
    for i in range(len(model_weight_of_u)):
        if model_weight_of_u[i] > max:
            max = model_weight_of_u[i]
            max_index = i
        # elif model_weight_of_u[i] == max:

    return max_index


def evaluate_adaboost_ensemble(models, model_weight_per_user, train_data, test_data, irt_wrong):
    total_array = np.zeros((542, 1))
    correct_array = np.zeros((542, 1))
    wrong_array = np.zeros((542, 1))
    wrong_per_model = np.zeros((542, 3))

    correct = 0
    total = 0
    wrong = 0

    nbrs = models[0]
    knn_train_data = train_data[0]
    mat = nbrs.fit_transform(knn_train_data)
    theta = models[1][0]
    beta = models[1][1]
    nn_model = models[2]
    nn_train_data = train_data[2]
    nn_train_data = torch.FloatTensor(nn_train_data)
    prediction = 0

    for i, u in enumerate(test_data["user_id"]):  # i=index u=u_id q=q_id
        print("correct=" + str(correct) + ";wrong=" + str(wrong) + ";total=" + str(total))
        most_suited_model = pick_suited_model(u, model_weight_per_user)
        q = test_data["question_id"][i]
        if most_suited_model == 0:
            # We use NaN-Euclidean distance measure.
            prediction = mat[u][q]
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0
            # x = (theta[u] - beta[q]).sum()
            # p_a = sigmoid(x)
            # if p_a < 0.5:
            #     prediction = 0
            # elif p_a >= 0.5:
            #     prediction = 1
        if most_suited_model == 1:
            x = (theta[u] - beta[q]).sum()
            p_a = sigmoid(x)
            if p_a < 0.5:
                prediction = 0
            elif p_a >= 0.5:
                prediction = 1
        if most_suited_model == 2:
            # nn_model.eval()
            #
            inputs = Variable(nn_train_data[u]).unsqueeze(0)  # get that user's train data
            output = nn_model(inputs)  # generate that user's prediction
            prediction = output[0][test_data["question_id"][i]].item() >= 0.5
            # x = (theta[u] - beta[q]).sum()
            # p_a = sigmoid(x)
            # if p_a < 0.5:
            #     prediction = 0
            # elif p_a >= 0.5:
            #     prediction = 1

        target = test_data["is_correct"][i]
        if prediction == target:
            correct += 1
            correct_array[u] += 1
        else:
            wrong += 1
            wrong_array[u] += 1
            wrong_per_model[u][most_suited_model] += 1
        total_array[u] += 1
        total += 1
    test_acc = correct / float(total)
    print("lets hope this get over 70:" + str(test_acc))
    return test_acc


def main():
    knn_param = []

    print(run_adaboost_ensemble())


if __name__ == "__main__":
    main()