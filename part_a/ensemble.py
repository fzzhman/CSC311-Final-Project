# TODO: complete this file.

from sklearn.impute import KNNImputer
import torch
from torch import optim
from torch.autograd import Variable
import neural_network as nn
import item_response as ir
from utils import *
import numpy as np


def train_neural_network(valid_data,zero_train_matrix, train_matrix, lr=0.05, l2lambda=0.03, epochs=10, k=10):
    print("Training NN with lr: {}, l2 lambda: {}, epoch: {}, k: {}".format(lr, l2lambda, epochs, k))

    U, Q = np.shape(zero_train_matrix)
    model = nn.AutoEncoder(num_question=Q, k=k, p=0)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr)
    for iteration in range(epochs):
        train_loss = 0
        for user_id in range(U):
            inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss += ((l2lambda * 0.5) * model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        #valid_acc = evaluate(model, zero_train_matrix, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epochs, train_loss, 0))
    model.eval()
    return model


def train_ir(train_data):
    # TODO: Implement once IRT is complete.
    pass


def predict_knn(sparse_user_data, question_id, train_matrix, k=10):
    knn_mat = torch.cat((train_matrix, sparse_user_data.unsqueeze(0)))
    nbrs = KNNImputer(n_neighbors=k)
    return nbrs.fit_transform(knn_mat)[-1, question_id]


def predict_nn(sparse_user_data, question_id, trained_model):
    o = trained_model(Variable(sparse_user_data).unsqueeze(0))
    return o[0][question_id].item()


def predict_ir():
    # TODO: Implement once IRT is complete.
    pass


def evaluate_ensemble(n_samples=300):
    zero_train_matrix, train_matrix, valid_data, test_data = nn.load_data()
    knn_base_dist = np.random.randint(0, len(train_matrix), n_samples)
    nn_base_dist = np.random.randint(0, len(train_matrix), n_samples)
    # TODO: Implement IRT

    # Train
    nn_model = train_neural_network(valid_data,zero_train_matrix[nn_base_dist], train_matrix)
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
    print(evaluate_ensemble(n_samples=400))


if __name__ == "__main__":
    main()
