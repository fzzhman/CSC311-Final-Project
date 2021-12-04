import sys

from matplotlib.pyplot import axis
from utils import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(sparse_matrix, theta, beta):
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
    sparse_matrix = sparse_matrix.toarray()
    num_users, num_questions = np.shape(sparse_matrix)
    theta_mat = np.tile(np.expand_dims(theta, 1), (1, num_questions))
    beta_mat = np.tile(np.expand_dims(beta, 0), (num_users, 1))

    log_lklihood = sparse_matrix * np.log(sigmoid(theta_mat - beta_mat)) + (1-sparse_matrix) * (1 - sigmoid(theta_mat - beta_mat))
    log_lklihood[np.isnan(log_lklihood)] = 0
    log_lklihood = np.sum(log_lklihood) # Only include positives, and sum instead of multiply (log)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(sparse_matrix, lr, theta, beta):
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
    num_users, num_questions = np.shape(sparse_matrix)

    c_mat = sparse_matrix.toarray()

    theta_mat = np.tile(np.expand_dims(theta, 1), (1, num_questions))
    beta_mat = np.tile(np.expand_dims(beta, 0), (num_users, 1))

    theta_grad = c_mat - sigmoid(theta_mat - beta_mat)
    theta_grad = -theta_grad # negative log likelihood
    theta_grad[np.isnan(theta_grad)] = 0
    theta_grad_collapsed = np.sum(theta_grad, 1)
    theta = theta - (lr * theta_grad_collapsed)

    beta_grad = -c_mat + sigmoid(theta_mat - beta_mat)
    beta_grad = -beta_grad # negative log likelihood
    beta_grad[np.isnan(beta_grad)]  = 0
    beta_grad_collapsed = np.sum(beta_grad, 0)
    beta = beta - (lr * beta_grad_collapsed)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta

def irt(train_sparse_mat, val_data, lr, iterations, verbosity = 1):
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
    num_users, num_questions = np.shape(train_sparse_mat)
    rng = np.random.default_rng()
    theta = np.asarray(rng.uniform(0.001, 1, num_users)) # Size is of number of users based on ID.
    beta = np.asarray(rng.uniform(0.001, 1, num_questions)) # Size is of number of questions based on question ID.

    # Convert validation data to matrix.
    val_sparse_mat = np.empty(np.shape(train_sparse_mat))
    val_sparse_mat[:] = np.nan
    for i, u_id in enumerate(val_data["user_id"]):
        q_id = val_data["question_id"][i]
        val_sparse_mat[u_id, q_id] = val_data["is_correct"][i]
    val_sparse_mat = sparse.csr_matrix(val_sparse_mat)


    train_nllks = []
    val_nllks = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(train_sparse_mat, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_nllks.append(train_neg_lld)

        val_neg_lld = neg_log_likelihood(val_sparse_mat, theta=theta, beta=beta)
        val_nllks.append(val_neg_lld)
        if verbosity == 2:
            print("It.: {} Train NLLK: {} Val. Score: {}".format(i, train_neg_lld, score))
        elif verbosity == 1:
            sys.stdout.write("\rIt.: {} Train NLLK: {} Val. Score: {}".format(i, train_neg_lld, score))
        theta, beta = update_theta_beta(train_sparse_mat, lr, theta, beta)
    if verbosity == 1:
        print()

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_nllks, val_nllks


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def main(data_dir = "../data"):
    # train_data = load_train_csv(data_dir)
    # You may optionally use the sparse matrix.
    train_sparse_matrix = load_train_sparse(data_dir)
    val_data = load_valid_csv(data_dir)
    test_data = load_public_test_csv(data_dir)


    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # iteration_sets = [45, 180, 185, 190]
    # lrs = [0.001, 0.005]

    iteration_sets = [45]
    lrs = [0.005]


    results = {
        "iteration_vals": [],
        "train_nllks": [],
        "val_nllks": [],
        "lr": 0,
        "iterations": 0,
        "val_acc": 0,
        "test_acc": 0,
        "theta": None,
        "beta": None
    }

    for iterations in iteration_sets:
        for lr in lrs:
            print("Training with lr of {} and {} number of iterations.".format(lr, iterations))
            theta, beta, train_nllks, val_nllks = irt(train_sparse_matrix, val_data, lr, iterations, verbosity=0)
            acc = evaluate(val_data, theta, beta)
            print("Final accuracy: {}".format(acc))
            if acc > results["val_acc"]:
                results["val_acc"] = acc
                results["lr"] = lr
                results["iterations"] = iterations
                results["train_nllks"] = train_nllks
                results["val_nllks"] = val_nllks
                results["theta"] = theta
                results["beta"] = beta
    
    results["iteration_vals"] = np.arange(results["iterations"])

    print("lr*: {} iterations*: {} val_acc*: {}".format(results["lr"], results["iterations"], results["val_acc"]))

    test_acc = evaluate(test_data, theta, beta)
    print("test accuracy: {}".format(test_acc))
    results["test_acc"] = test_acc
    
    plt.figure()
    plt.plot(results["train_nllks"], label="train")
    plt.plot(results["val_nllks"], label="valid")
    plt.ylabel("Negative Log Likelihood")
    plt.xlabel("Iteration")
    plt.title("Neg Log Likelihood for Train and Validation Data")
    plt.legend()
    plt.show()
    plt.savefig("figures/generated/irt_{}.pdf".format("Neg Log Likelihood for Train and Validation Data".replace(" ", "_")))

    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    questions = [1, 2, 3]
    plt.figure()
    sorted_theta = results["theta"]
    sorted_theta.sort()
    for q in questions:
        q_str = "q. {}".format(q)
        beta_q = results["beta"][q]
        plt.plot(sigmoid(sorted_theta - beta_q), label=q_str)
    plt.ylabel("Probabilty of the Correct Response")
    plt.xlabel("Theta")
    plt.title("Probability User Answer Correct w.r.t Theta")
    plt.legend()
    plt.show()
    plt.savefig("figures/generated/irt_{}.pdf".format("Probability of User Answering Correctly vs Theta".replace(" ", "_")))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return results


if __name__ == "__main__":
    main()
