import sys

from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
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
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encoder = nn.Sequential(
            self.g,
            nn.Sigmoid(),
        )
        decoder = nn.Sequential(
            self.h,
            nn.Sigmoid()
        )

        encoded = encoder(inputs)
        decoded = decoder(encoded)
        out = decoded
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, num_epoch, valid_data=None, verbosity=1):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param num_epoch: int
    :param valid_data: Dict
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    loss_array = []
    valid_acc_array = []
    epoch_array = []
    for epoch in range(0, num_epoch):
        train_loss = 0.
        epoch_array.append(epoch)
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss += (lamb/2)*model.get_weight_norm()  # L2 Regularization
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        if not valid_data == None:
            loss_array.append(train_loss)
            valid_acc = evaluate(model, zero_train_data, valid_data)
            valid_acc_array.append(valid_acc)
            if verbosity == 2:
                print("{}/{}\t{:.6f}\t{}".format((epoch + 1), num_epoch, train_loss, valid_acc))
            elif verbosity == 1:
                sys.stdout.write(
                    "\rEpoch: {}/{}, Loss: {:.6f}, Valid acc.: {}".format((epoch + 1), num_epoch, train_loss,
                                                                          valid_acc))
    print("\nFinal training results:")
    print("\rEpoch: {}/{}, Loss: {:.6f}, Valid acc.: {}".format((epoch + 1), num_epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return epoch_array, valid_acc_array, loss_array

def evaluate(model, train_data, valid_data):
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

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def accuracy_plot(epochs, accuracy, title):
    """Generate plot for user-based collaborative filtering;
    validation data accuracy against k value

    :param: k_vals: List of k values for this knn function
    :param: accuracy: list of accuracy values generated by knn function
    :param: title: String Value Graph Title"""

    plt.plot(epochs, accuracy)
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.title(title)
    plt.show()


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data("../data")

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    ks = [10,]
    lrs = [ 0.05]
    num_epochs = [22]
    lambs = [ 0,0.00025,0.1]
    model = None

    u, q = train_matrix.shape
    acc_star = 0
    k_star = 0
    lr_star = 0
    num_epoch_star = 0
    lamb_star = 0
    for k in ks:
        print("Starting k, learning rate, number of epochs, and lambda optimizations.")

        for lr in lrs:
            for num_epoch in num_epochs:
                for lamb in lambs:
                    print("\tk\tLearn. Rate\t# of Epochs\tLambda")
                    print("\t{}\t{}\t{}\t{}".format(k, lr, num_epoch, lamb))
                    model = AutoEncoder(q, k)
                    epoch_array, valid_acc_array, loss_array = \
                        train(model, lr, lamb, train_matrix, zero_train_matrix, num_epoch, valid_data, verbosity=2)
                    acc = evaluate(model, zero_train_matrix, valid_data)
                    test_acc = evaluate(model, zero_train_matrix, test_data)
                    accuracy_plot(epoch_array,valid_acc_array, "Validation Accuracy Change As Epoch Increases")
                    accuracy_plot(epoch_array, loss_array, "Loss Change As Epoch Increases")
                    if acc > acc_star:
                        print("\tFinal Validation Accuracy*\t{}".format(acc))
                        print("\tFinal Test Accuracy*\t{}".format(test_acc))
                        acc_star = acc
                        k_star = k
                        lr_star = lr
                        num_epoch_star = num_epoch
                        lamb_star = lamb
                    else:
                        print("\tFinal accuracy\t{}".format(acc))
                        print("\tFinal Test Accuracy*\t{}".format(test_acc))
        print("Concluding learning rate, number of epoch and lambda optimizations.")
        print("Results:")
        print("\tFinal acc.:\t{}".format(acc_star))
        print("\tk*:\t{}".format(k_star))
        print("\tLearning rate*:\t{}".format(lr_star))
        print("\tNumber of epochs*:\t{}".format(num_epoch_star))
        print("\tLambda*:\t{}".format(lamb_star))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
