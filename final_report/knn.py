from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


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
    # print("Validation Accuracy: {}".format(acc))
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
    nbrs = KNNImputer(n_neighbors=k)
    matrix = matrix.T
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    # print("Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def accuracy_plot(k_vals, accuracy, title):
    """Generate plot for user-based collaborative filtering;
    validation data accuracy against k value

    :param: k_vals: List of k values for this knn function
    :param: accuracy: list of accuracy values generated by knn function
    :param: title: String Value Graph Title"""

    plt.figure()
    plt.plot(k_vals, accuracy)
    plt.xlabel("k_values")
    plt.ylabel("Accuracy on Validation Data")
    plt.xticks(k_vals)
    plt.title(title)
    plt.show()
    if os.path.exists("figures/generated/"):
        plt.savefig("figures/generated/knn_{}.pdf".format(title.replace(" ", "_")))


def main(data_path = "../data", verbosity = 1):
    sparse_matrix = load_train_sparse(data_path).toarray()
    val_data = load_valid_csv(data_path)
    test_data = load_public_test_csv(data_path)

    if verbosity > 1:
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

    k_vals = [1, 6, 11, 16, 21, 26]
    
    val_user_acc = []
    for k in k_vals:
        val_user_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))
    accuracy_plot(k_vals, val_user_acc, "User-Based Collaborative Filtering")
    max_k_idx_user = val_user_acc.index(max(val_user_acc))
    max_k_user = k_vals[max_k_idx_user]
    if verbosity > 0:
        print("k*: {} val. acc.: {}".format(max_k_user, max(val_user_acc)))

    val_item_acc = []
    for k in k_vals:
        val_item_acc.append(knn_impute_by_item(sparse_matrix, val_data, k))
    accuracy_plot(k_vals, val_item_acc, "Item-Based Collaborative Filtering")
    max_k_idx_item = val_item_acc.index(max(val_item_acc))
    max_k_item = k_vals[max_k_idx_item]
    if verbosity > 0:
        print("k*: {} val. acc.: {}".format(max_k_item, max(val_item_acc)))

    # part (d)

    user_test = knn_impute_by_user(sparse_matrix, test_data, max_k_user)
    item_test = knn_impute_by_item(sparse_matrix, test_data, max_k_item)

    print("User-Based Algorithm Accuracy on Test Data: ", user_test)
    print("Item-Based Algorithm Accuracy on Test Data: ", item_test)
    if user_test > item_test:
        print("User-based collaborative filtering works better")
    elif item_test > user_test:
        print("Item-based collaborative filtering works better")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return k_vals, val_user_acc, val_item_acc

if __name__ == "__main__":
    main()
