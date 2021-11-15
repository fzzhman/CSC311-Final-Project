from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    #####################################################################
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy (question): {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    accuracies_user = []
    
    for i in k_values:
        accuracies_user.append(knn_impute_by_user(sparse_matrix, val_data, i))
    k_user = accuracies_user.index(max(accuracies_user))
    chosen_k = k_values[k_user]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, chosen_k)
    print("Highest k value for knn_impute_by_user: ", chosen_k)
    print("Test Accuracy for this value of k: ", test_acc)
    fig1 = plt.figure()
    ax = fig1.add_axes([0, 0, 1, 1])
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    plt.plot(k_values, accuracies_user, "-g", label="validation")
    plt.legend(loc="upper right")
    plt.show()
    
    accuracies_ques = []
    for i in k_values:
        accuracies_ques.append(knn_impute_by_item(sparse_matrix, val_data, i))
    k_ques = accuracies_ques.index(max(accuracies_ques))
    chosen_ques_k = k_values[k_ques]
    test_acc_ques = knn_impute_by_item(sparse_matrix, test_data, chosen_ques_k)
    print("Highest k value for knn_impute_by_item: ", chosen_ques_k)
    print("Test Accuracy for this value of k: ", test_acc_ques)
    fig1 = plt.figure()
    ax = fig1.add_axes([0, 0, 1, 1])
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    plt.plot(k_values, accuracies_ques, "-g", label="validation")
    plt.legend(loc="upper right")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
