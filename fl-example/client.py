import warnings
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import utils

if __name__ == "__main__":
    # Load dataset
    X_train = np.loadtxt('train_X.csv')    
    y_train = np.loadtxt('train_y.csv')

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=50,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class FLClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_validation, model.predict_proba(X_validation))
            y_score = model.predict_proba(X_validation)[:, 1]
            auc = roc_auc_score(y_validation, y_score)
            return loss, len(X_validation), {"auc": auc}
            


    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FLClient())
