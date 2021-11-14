# TODO: complete this file.
from utils import *
from random import *
import neural_network

"""
Ensemble. In this problem, you will be implementing bagging ensemble to im-
prove the stability and accuracy of your base models.

Select and train 3 base models with
bootstrapping the training set.

You may use the same or different base models. Your imple-
mentation should be completed in part_a/ensemble.py.

To predict the correctness, generate
3 predictions by using the base model and average the predicted correctness. Report the final
validation and test accuracy. Explain the ensemble process you implemented. Do you obtain
better performance using the ensemble? Why or why not?
"""

"""
train 3 basemodels
"""
def train_base_model():
    """

    :return:
    """
    zero_train_matrix, train_matrix, valid_data, test_data = neural_network.load_data()
    base_model = []
    """
    kNN
    """

    """
    Item Response Theory
    """


    """
    neural network
    """
    # Set model hyperparameters.
    k_star = 100  # placeholder 100
    model = neural_network.AutoEncoder(num_question=len(train_matrix[0]), k=k_star)

    # Set optimization hyperparameters.
    lr = 0.01  # learning rate
    num_epoch = 10  # 1 = go through all train data for 1 time
    lamb = None  # not used?

    base_model[2] = neural_network.train(model, lr, lamb, train_matrix, zero_train_matrix,
                                         valid_data, num_epoch)

    return base_model





"""
given sample x and basemodel 
generate prediction of a basemodel
"""
def generate_prediction(x, base_model):

    y = np.zeros(3)

    y[2] = base_model[2].evaluate(x)

    avg_y = np.mean(y)

    return avg_y

"""
sample with replacement
"""
def bootstrap(data, num_batches, batchsize):
    data_len = len(data)

    all_batches = np.zeros(num_batches)

    for s in range(batchsize):
        random = randrange(data_len)
        current_batch = np.zeros(batchsize)

        for n in range(batchsize):
            current_batch[n] = data[random]

        all_batches[s] = current_batch

    return all_batches

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

def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")


    base_models = train_base_model()
    predictions = np.mean(generate_prediction(base_model))

if __name__ == "__main__":
    main()