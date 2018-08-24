import preprocessing_module as preproc
from torch.utils.data import DataLoader
from visdom import Visdom
from util.visual import VisualManager
import torchvision as tv
from custom_transforms import ToBand
from util.parser import Parser
from util.io import *

args = \
[
[('--model', '-m'), {'type': str, 'default': 'lenet', 'help': 'Used Model lenet | allcnn |'}],
[('--lr',), {'type': float, 'default': 0.01, 'help': 'Learning rate'}],
[('-d',), {'type': float, 'default': 1, 'help': 'Dropout probability'}],
[('--dataset',), {'type': str, 'default': 'mnist', 'help': 'Dataset to use'}],
[('--root',), {'type': str, 'default': '/mnt/DATA/TorchData', 'help': 'Location of the dataset'}],
[('--save_path', '-s'), {'type': str, 'default': '/mnt/DATA/ProjectsResults/contestai', 'help': 'Results path'}],
[('--batch_size', '-bs'), {'type': int, 'default': 128, 'help': 'Batch size'}],
[('--epochs', '-e'), {'type': int, 'default': 100, 'help': 'Number of epochs'}],
[('--log_period', '-lp'), {'type': int, 'default': 30, 'help': 'Logging period in number of epochs'}],
[('--optimizer', '-opt'), {'type': str, 'default': 'adam', 'help': 'Optimizer to use'}],
]

argparser = Parser("Deep Elliptical Embeddings")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

dirname = build_dirname(opt, ('lr', 'batch_size', 'optimizer'))
savepath = make_save_directory(opt, dirname)

vis = Visdom()
vm = VisualManager(vis, dirname)

W = 1280
H = 180

path_training_ok = '/mnt/DATA/beantech_contestAI/Dataset2/campioni OK'
path_training_ko = '/mnt/DATA/beantech_contestAI/Dataset2/campioni KO'
path_validation_ok = '/mnt/DATA/beantech_contestAI/Dataset1/campioni OK'
path_validation_ko = '/mnt/DATA/beantech_contestAI/Dataset1/campioni KO'

num_epochs = 1
dataset_name = 'albedo'

training_set = preproc.build_datasets(path_training_ok, path_training_ko, dataset_name)
training_loader = DataLoader(training_set, batch_size=1, shuffle=True)

#validation_set = preproc.build_datasets(path_validation_ok, path_validation_ko, dataset_name)

#%%
training_set.transform = tv.transforms.Compose([ToBand(H, margin=2), tv.transforms.ToTensor()])

#%%
for epoch in range(opt['epochs']):
    for batch in training_loader:
        # training
        print('Training code here')




