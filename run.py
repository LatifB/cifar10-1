import torch
import torch.optim as optim

import argparse
import json

from src.Model import Net
from src.DataPrep import DataPrep
from src.Optimizer import OptunaOptimizer

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--optimize', action='store_true')
ap.add_argument('-t', '--train', action='store_true')
options = ap.parse_args()
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_loaders = DataPrep()
    train_loader = data_loaders.get_train_loader()
    valid_loader = data_loaders.get_valid_loader()
    test_loader = data_loaders.get_test_loader()
    train_mini_loader = data_loaders.get_train_mini_loader()
    vald_mini_loader = data_loaders.get_vald_mini_loader()

    model = Net()

    if options.optimize:
        optuna_opt = OptunaOptimizer(Net, train_mini_loader, vald_mini_loader, epochs=5, device=device)
        params = optuna_opt.best_params
    else:
        with open('./src/params.json') as f:
            params = json.load(f)

    if options.train:
        if params['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']), eps=params['epsilon'], weight_decay=params['weight_decay'])
        elif params['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']), eps=params['epsilon'], weight_decay=params['weight_decay'])
        elif params['optimizer'] == 'ASGD':
            optimizer = optim.ASGD(model.parameters(), lr=params['lr'], lambd=params['lambd'], alpha=params['alpha'], t0=params['t0'], weight_decay=params['weight_decay'])
        elif params['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'], dampening=params['dampening'], weight_decay=params['weight_decay'])

        model.fit(train_loader, optimizer, 40, device=device, valid_loader=valid_loader, verbose=3)
        model.save()
    else:
        model.load_state_dict(torch.load('cifar10-1.pth', map_location=torch.device(device)))

    model.test(test_loader, device=device)
