
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


def weighted_bagging(train_matrix, weight_matrix, N):
    bagging_order = weight_matrix.copy()
    current_training_set = train_matrix.copy()
    rng = np.random.default_rng()  # get a default random generator

    for i in range(N):
        n = rng.uniform(0, 1) #get a number between 1 and zero
        current_bottom = 0

        for sample_index in range(len(weight_matrix)):
            current_top = current_bottom + weight_matrix[sample_index]
            if current_bottom <= n <= current_top:
                current_training_set[i] = train_matrix[sample_index]
                bagging_order[i] = sample_index
                break
            current_bottom = current_top

    current_zero_training_set = current_training_set.copy()
    current_zero_training_set[np.isnan(train_matrix)] = 0
    return current_training_set, current_zero_training_set, bagging_order


def evaluate_model_weight(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    got_wrong = np.zeros((len(train_data),1))
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        else:
            got_wrong[u] += 1
        total += 1
    acc = correct / float(total)
    model_weight = 0.5 * np.log(acc / (1 - acc))

    return model_weight,got_wrong


def update_sample_weight(sample_weight_matrix,wrong_samples, model_weight):
    updated_weight_sum = 0
    for item_index in range(len(wrong_samples)):

        if wrong_samples[item_index] > 0: # samples that got wrong need more focus, increase weight
            sample_weight_matrix[item_index] = sample_weight_matrix[item_index] * (np.e ** model_weight)
        elif wrong_samples[item_index] <= 0: #samples that got correct can predict easily, decrease weight
            sample_weight_matrix[item_index] = sample_weight_matrix[item_index] * (np.e ** -model_weight)

        updated_weight_sum += sample_weight_matrix[item_index]
    sample_weight_matrix = sample_weight_matrix / updated_weight_sum
    
    return sample_weight_matrix


def evaluate_adaboost_ensemble():

    # set initial parameters, load data
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    N = len(zero_train_matrix)
    initial_weight = 1 / N
    sample_weight_matrix = np.full((N, 1), initial_weight)

    model_weights = np.ones((3,1))

    for model_index in range(3):
        # bootstrap
        current_training_set,current_zero_training_set,bagging_order = \
            weighted_bagging(train_matrix, sample_weight_matrix, N)

        current_training_set = torch.FloatTensor(current_training_set)
        train_matrix = torch.FloatTensor(train_matrix)
        current_zero_training_set = torch.FloatTensor(current_zero_training_set)
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)

        model = 0
        # train
        if model_index == 0:
        # train knn
            model = nn.AutoEncoder(num_question=len(train_matrix[0]), k=11, p=0)
            print("training neural net")
            print("k* = " + str(11) + "; learning rate = " + str(0.01)
                  + "; num_epoch = " + str(20) + "; lamb=" + str(0.00025) + "; p=" + str(0))
            nn.train(model,
                     lr=0.01,
                     lamb=0.00025,
                     train_data=train_matrix,
                     zero_train_data=zero_train_matrix,
                     valid_data=valid_data,
                     num_epoch=20)
            test_acc = nn.evaluate(model, current_training_set, test_data)
            print("Test Acc: {}".format(test_acc))
        elif model_index == 1:
        # train IRT
            model = nn.AutoEncoder(num_question=len(train_matrix[0]), k=12, p=0)
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
        model_weights[model_index],wrong_samples = evaluate_model_weight(model,
                                                                        current_training_set,
                                                                        valid_data=valid_data)
        # update sample weight
        sample_weight_matrix = update_sample_weight(sample_weight_matrix,wrong_samples,model_weights[model_index])


def main():
    print(evaluate_adaboost_ensemble())


if __name__ == "__main__":
    main()