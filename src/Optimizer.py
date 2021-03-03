import torch.optim as optim
import optuna
import json


class OptunaOptimizer():

    def __init__(self, model, train_loader, vald_loader, n_trials=100, epochs=1, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.vald_loader = vald_loader
        self.n_trials = n_trials
        self.epochs = epochs
        self.device = device
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=self.n_trials)
        self.best_params = self.study.best_params
        self.write_to_json(self.best_params)

    def objective(self, trial):
        model = self.model()

        optimizer = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'ASGD', 'SGD'])
        lr = trial.suggest_loguniform('lr', 1e-7, 1e-3)

        if optimizer == 'Adam':
            beta1 = trial.suggest_float('beta1', 0.7, 1)
            beta2 = trial.suggest_float('beta2', 0.7, 1)
            weight_decay = trial.suggest_float('weight_decay', 0, 1e-1)
            epsilon = trial.suggest_float('epsilon', 0, 1e-5)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
        elif optimizer == 'AdamW':
            beta1 = trial.suggest_float('beta1', 0.7, 1)
            beta2 = trial.suggest_float('beta2', 0.7, 1)
            epsilon = trial.suggest_float('epsilon', 0, 1e-5)
            weight_decay = trial.suggest_float('weight_decay', 0, 1e-1)
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
        elif optimizer == 'ASGD':
            lambd = trial.suggest_float('lambd', 0, 1e-6)
            alpha = trial.suggest_float('alpha', 0.5, 1)
            t0 = trial.suggest_float('t0', 0, 1e-4)
            weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-1)
            optimizer = optim.ASGD(model.parameters(), lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        elif optimizer == 'SGD':
            momentum = trial.suggest_float('momentum', 0.7, 1)
            weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-1)
            dampening = trial.suggest_float('dampening', 0, 1e-1)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)

        accuracy = model.fit(self.train_loader, optimizer, self.epochs, device=self.device, valid_loader=self.vald_loader, verbose=0)
        del model

        return accuracy

    def write_to_json(self, params):
        with open('./src/params.json', 'w') as json_file:
            json.dump(params, json_file)
