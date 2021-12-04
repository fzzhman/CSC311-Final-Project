from utils import *
import random
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_likelihood = 0.

    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        correct = data["is_correct"][i]

        x = theta[user] - beta[question]

        log_likelihood += correct*(np.log(sigmoid(x)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_likelihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.
    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_i_arr = np.array(data["user_id"])
    question_j_arr = np.array(data["question_id"])
    correct_ij_arr = np.array(data["is_correct"])

    theta_copy = theta.copy()
    beta_copy = beta.copy()
    for i in range(len(theta)):
        theta[i] -= lr * (np.sum(sigmoid(theta_copy[i] - beta_copy)[question_j_arr[user_i_arr == i]])
                          - np.sum(correct_ij_arr[user_i_arr == i]))

    for j in range(len(beta)):
        beta[j] -= lr * (np.sum(correct_ij_arr[question_j_arr == j]) -
                         np.sum(sigmoid(theta_copy - beta_copy[j])[user_i_arr[question_j_arr == j]]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    np.random.seed(1)
    theta = np.random.rand(542)  # init theta randomly
    beta = np.random.rand(1774)  # init beta randomly

    val_log_likelihood, train_log_likelihood = [], []

    for i in range(iterations):
        # Log likelihood
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_log_likelihood.append(train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_likelihood.append(val_neg_lld)

        train_score = evaluate(data=data, theta=theta, beta=beta)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        #
        print("NLLK: {} \t Train Score: {} \t Validation Score: {}"
              .format(train_neg_lld, train_score, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

        #  TODO: You may change the return values to achieve what you want.
    return theta, beta, val_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    predictions = []
    for i, j in enumerate(data["question_id"]):
        users = data["user_id"][i]
        x = (theta[users] - beta[j]).sum()
        prediction_a = sigmoid(x)
        predictions.append(prediction_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(predictions))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr, iterations = 0.009, 20
    theta, beta, val_log_likelihood, train_log_likelihood= \
        irt(train_data, val_data, lr, iterations)
    val_score = evaluate(data=val_data, theta=theta, beta=beta)
    test_score = evaluate(data=test_data, theta=theta, beta=beta)
    print("Chosen lr parameter: ", lr)
    print("Chosen number of iterations: ", iterations)

    print("Validation Accuracy: ", val_score)
    print("Test Accuracy: ", test_score)

    #####################################################################
    # Part (b) Plots
    plt.plot(train_log_likelihood, label="train")
    plt.plot(val_log_likelihood, label="valid")
    plt.ylabel("Negative Log Likelihood")
    plt.xlabel("Iteration")
    plt.xticks(np.arange(0, iterations, 1))
    plt.title("Neg Log Likelihood for Train and Validation Data")
    plt.legend()
    plt.show()
    #########################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    np.random.seed(212)
    questions = random.sample(val_data["question_id"], 3)
    # randomly chose 3 questions from validation data
    theta = theta.reshape(-1)
    theta.sort()
    for q in questions:
        plt.plot(theta, sigmoid(theta - beta[q]),
                 label=f"Question {q}")
    plt.ylabel("Probability")
    plt.xlabel("Theta")
    plt.title("Selected Questions Probability as a function of Theta")
    plt.legend()
    plt.show()
    plt.savefig("abc.pdf")
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

