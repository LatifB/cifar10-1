## Cifar10-1
This is a trial with PyTorch on cifar10 dataset. Hyperparameters are found with optuna and the best hyperparameters are stores in '/src/params.json' file. After training with 40 epochs model achieved 84.44% accuracy on the test dataset.
Architecture of the model is conv -> batch normalization -> conv -> batch normalization -> maxpool -> conv -> batch normalization -> maxpool -> flatten -> dense -> bach normalization -> dense -> bach normalization -> dense -> bach normalization -> dense -> bach normalization -> dense -> bach normalization -> dense.

# How to run locally
```bash
git clone git@github.com:LatifB/cifar10-1.git
cd cifar10-1
pip install -r requirements.txt
python run.py -t
```
Additionally, you can also run
```bash
python run.py -o -t
```
to also run hyperparameter optimization.

# How to run on colab
Repository also contains 'notebook.ipynb' file. This file can be uploaded to colab and run.

## License
[MIT](https://choosealicense.com/licenses/mit/)