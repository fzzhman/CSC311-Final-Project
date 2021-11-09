# TODO: complete this file.
from utils import *
from random import *
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
def train_base_model(data, label, model):

    return base_model

"""
given sample x and basemodel 
generate prediction of a basemodel
"""
def generate_prediction(x, base_model):

    return y

"""
sample with replacement
"""
def bootstrap(data, expected_groups, expected_group_sizes):
    data_len = len(data)

    all_groups = np.zeros(expected_groups)

    for s in range(expected_group_sizes):
        random = randrange(data_len)
        current_group = np.zeros(expected_group_sizes)

        for n in range(expected_group_sizes):
            current_group[n] = data[random]

        all_groups[s] = current_group

    return all_groups

def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")


    base_model = train_base_model()
    generate = np.mean(generate_prediction(base_model))

if __name__ == "__main__":
    main()