import preprocessing_module as preproc
from torch.utils.data import DataLoader
from visdom import Visdom
from visualmanager import *
import torchvision as tv
from importlib import reload
import numpy as np
from custom_transforms import ToBand
from util.parser import Parser
args = \
[
[('--model', '-m'), {'type': str, 'default': 'lenet', 'help': 'Used Model lenet | allcnn |'}],
[('--margin', '-mg'), {'type': float, 'default': 5.0, 'help': 'Margin used for the hinge loss'}],
[('--dim',), {'type': int, 'default': 2, 'help': 'Dimension of the embedding'}],
[('--lr',), {'type': float, 'default': 0.01, 'help': 'Learning rate'}],
[('-d',), {'type': float, 'default': 1, 'help': 'Dropout probability'}],
[('--dataset',), {'type': str, 'default': 'mnist', 'help': 'Dataset to use'}],
[('--root',), {'type': str, 'default': '/mnt/DATA/TorchData', 'help': 'Location of the dataset'}],
[('--save_path', '-s'), {'type': str, 'default': '/mnt/DATA/ProjectsResults/EllipticalEmb', 'help': 'Results path'}],
[('--num_positive', '-np'), {'type': int, 'default': 1, 'help': 'Number of same class elements per training sample'}],
[('--num_negative', '-nn'), {'type': int, 'default': 1, 'help': 'Number of different class elements per training sample'}],
[('--batch_size', '-bs'), {'type': int, 'default': 128, 'help': 'Batch size'}],
[('--epochs', '-e'), {'type': int, 'default': 30, 'help': 'Number of epochs'}],
[('--categories', '-c'), {'type': int, 'nargs': '+', 'default': None, 'help': 'Categories used to train the model'}],
[('--log_period', '-lp'), {'type': int, 'default': 30, 'help': 'Logging period in number of epochs'}],
[('--num_iters', '-ni'), {'type': int, 'default': 20, 'help': 'Number of iterations for the matrix sqrt algorithm'}],
[('--reg', ), {'type': float, 'default': 2.0, 'help': 'Regularization for the matrix sqrt algorithm'}],
[('--reg2', ), {'type': float, 'default': 1E-8, 'help': 'Regularization for the gradient of the bures metric'}],
]

argparser = Parser("Deep Elliptical Embeddings")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

#%%
W = 1280
H = 180
#%%
path_training_ok = '/mnt/DATA/beantech_contestAI/Dataset2/campioni OK'
path_training_ko = '/mnt/DATA/beantech_contestAI/Dataset2/campioni KO'
path_validation_ok = '/mnt/DATA/beantech_contestAI/Dataset1/campioni OK'
path_validation_ko = '/mnt/DATA/beantech_contestAI/Dataset1/campioni KO'

num_epochs = 1
dataset_name = 'albedo'
images_type = 'OK'
training_set_ok = preproc.process_original_data(path_training_ok, dataset_name, images_type, transform=None)
training_generator_ok = DataLoader(training_set_ok, batch_size=1, shuffle=True)

#%%
training_set_ok.transform = tv.transforms.Compose([ToBand(H, margin=2), tv.transforms.ToTensor()])
viz = Visdom()
check_connection(viz)


data_iter = iter(training_generator_ok)
image = next(data_iter)
viz.image(image['image'], env='marco')

#%%
for epoch in range(num_epochs):
    for batch in training_generator_ok:
        # training
        print('Training code here')

exit(0)

