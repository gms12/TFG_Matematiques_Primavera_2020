from typing import Dict, Tuple, Optional, Iterable
import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import spatial

import torch
import torch.nn as nn
import torch.optim as optim

# Define the AE architecture
class NNAutoencoder(nn.Module):
    def __init__(self, input_shape, encode_shape):
        super().__init__()
        self._encode_layer = nn.Linear(in_features=input_shape, out_features=encode_shape)
        self._decode_layer = nn.Linear(in_features=encode_shape, out_features=input_shape)
        self._encode_shape = encode_shape

    def encode(self, X):
        return torch.tanh(self._encode_layer(X))

    def decode(self, X):
        return torch.tanh(self._decode_layer(X))

    def forward(self, X):
        return self.decode(self.encode(X))

    def predict(self, X):
        return self.forward(X).clone().detach()

    def train(self, X, y, epochs, optimizer, criterion, plot=False, **kwargs):
        losses = []
        im = []
        for i in range(epochs):
            optimizer.zero_grad()

            y_pred = self.forward(X)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return losses

class DashboardData():
    """
    Data class that handles the data used in the Dashboard.
    """

    def __init__(self):
        logging.info('Loading time series data...')
        self._data = np.load('dashboard/data/time_series_data.npy')
        logging.info('...data loaded')

        logging.info('Loading models...')
        self._models_st = self._load_models(tr_type='st')
        self._models_mm = self._load_models(tr_type='mm')
        logging.info('...models loaded')

        # We create the scalers
        logging.info('Creating scalers...')
        self._scaler_mm = MinMaxScaler()
        self._scaler_mm.fit(self._data.T)
        self._normalized_data = self._scaler_mm.transform(self._data.T).T

        self._scaler_st = StandardScaler()
        self._scaler_st.fit(self._data.T)
        self._standard_data = self._scaler_st.transform(self._data.T).T
        logging.info('...scalers created')

        # Load predictions
        logging.info('Loading predictions...')
        self._pred_st = self._load_predictions(tr_type='st')
        self._pred_mm = self._load_predictions(tr_type='mm')
        logging.info('...predictioncs loaded')

        # We transform the data into tensors
        self._X_train_st = torch.from_numpy(self._standard_data).type(torch.FloatTensor)
        self._X_train_mm = torch.from_numpy(self._normalized_data).type(torch.FloatTensor)

        # We get the hidden spaces
        logging.info('Getting hidden spaces...')
        self._hidden_st = self._hidden(models=self._models_st, test=self._X_train_st)
        self._hidden_mm = self._hidden(models=self._models_mm, test=self._X_train_mm)
        logging.info('...hidden spaces done')

        # We get the neighbour density
        logging.info('Computing neighbour density...')
        self._tree_st, self._nd_st = self._count_neighbours(hidden=self._hidden_st)
        self._tree_mm, self._nd_mm = self._count_neighbours(hidden=self._hidden_mm)
        logging.info('...neighbour density computed')

    @property
    def models_st(self):
        return self._models_st

    @property
    def models_mm(self):
        return self._models_mm

    @property
    def pred_st(self):
        return self._pred_st

    @property
    def pred_mm(self):
        return self._pred_mm

    @property
    def data(self):
        return self._data

    @property
    def hidden_st(self):
        return self._hidden_st

    @property
    def hidden_mm(self):
        return self._hidden_mm

    @property
    def nd_st(self):
        return self._nd_st

    @property
    def nd_mm(self):
        return self._nd_mm

    @property
    def tree_st(self):
        return self._tree_st

    @property
    def tree_mm(self):
        return self._tree_mm

    @property
    def data(self):
        return self._data

    @property
    def standard_data(self):
        return self._standard_data

    @property
    def normalized_data(self):
        return self._normalized_data

    def _hidden(self, models, test):
        hidden = {}
        for dim in models:
            hidden[dim] = models[dim].encode(test).clone().detach().numpy()
        return hidden

    def _count_neighbours(self, hidden):
        r = 0.1
        nc = {}
        trees = {}
        for i in [1, 2, 3]:
            nc[i] = []
            trees[i] = spatial.cKDTree(hidden[i])
            for point in hidden[i]:
                nc[i].append(len(trees[i].query_ball_point(point, r)) - 1)
        return trees, nc


    @staticmethod
    def _load_models(tr_type: str):
        input_shape = 365

        model_1 = NNAutoencoder(input_shape=input_shape, encode_shape=1)
        model_1.load_state_dict(torch.load('dashboard/models/ae_1_{}'.format(tr_type)))
        model_2 = NNAutoencoder(input_shape=input_shape, encode_shape=2)
        model_2.load_state_dict(torch.load('dashboard/models/ae_2_{}'.format(tr_type)))
        model_3 = NNAutoencoder(input_shape=input_shape, encode_shape=3)
        model_3.load_state_dict(torch.load('dashboard/models/ae_3_{}'.format(tr_type)))
        model_5 = NNAutoencoder(input_shape=input_shape, encode_shape=5)
        model_5.load_state_dict(torch.load('dashboard/models/ae_5_{}'.format(tr_type)))
        model_10 = NNAutoencoder(input_shape=input_shape, encode_shape=10)
        model_10.load_state_dict(torch.load('dashboard/models/ae_10_{}'.format(tr_type)))
        model_30 = NNAutoencoder(input_shape=input_shape, encode_shape=30)
        model_30.load_state_dict(torch.load('dashboard/models/ae_30_{}'.format(tr_type)))

        return {
            1: model_1, 2: model_2, 3: model_3,
            5: model_5, 10: model_10, 30: model_30,
        }

    @staticmethod
    def _load_predictions(tr_type: str):
        pred = np.load('dashboard/data/predictions_{}.npy'.format(tr_type))
        return {
            1: pred[0], 2: pred[1], 3: pred[2],
            5: pred[3], 10: pred[4], 30: pred[5],
        }
