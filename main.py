import preprocessing_module as preproc
from torch.utils.data import DataLoader
from visdom import Visdom
from util.visual import VisualManager
import torchvision as tv
from custom_transforms import ToBand
from util.parser import Parser

args = \
[
[('--model', '-m'), {'type': str, 'default': 'lenet', 'help': 'Used Model lenet | allcnn |'}],
[('--lr',), {'type': float, 'default': 0.01, 'help': 'Learning rate'}],
[('-d',), {'type': float, 'default': 1, 'help': 'Dropout probability'}],
[('--dataset',), {'type': str, 'default': 'mnist', 'help': 'Dataset to use'}],
[('--root',), {'type': str, 'default': '/mnt/DATA/TorchData', 'help': 'Location of the dataset'}],
[('--save_path', '-s'), {'type': str, 'default': '/mnt/DATA/ProjectsResults/ResDir', 'help': 'Results path'}],
[('--batch_size', '-bs'), {'type': int, 'default': 128, 'help': 'Batch size'}],
[('--epochs', '-e'), {'type': int, 'default': 100, 'help': 'Number of epochs'}],
[('--log_period', '-lp'), {'type': int, 'default': 30, 'help': 'Logging period in number of epochs'}],
]

argparser = Parser("Deep Elliptical Embeddings")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

W = 1280
H = 180

path_training_ok = '/mnt/DATA/beantech_contestAI/Dataset2/campioni OK'
path_training_ko = '/mnt/DATA/beantech_contestAI/Dataset2/campioni KO'
path_validation_ok = '/mnt/DATA/beantech_contestAI/Dataset1/campioni OK'
path_validation_ko = '/mnt/DATA/beantech_contestAI/Dataset1/campioni KO'

num_epochs = 1
dataset_name = 'albedo'

training_set_ok = preproc.process_original_data(path_training_ok, dataset_name, 1, transform=None)
training_generator_ok = DataLoader(training_set_ok, batch_size=1, shuffle=True)

training_set_ko = preproc.process_original_data(path_training_ok, dataset_name, 1, transform=None)
training_generator_ko = DataLoader(training_set_ok, batch_size=1, shuffle=True)
#%%
training_set_ok.transform = tv.transforms.Compose([ToBand(H, margin=2), tv.transforms.ToTensor()])
vis = Visdom()
vm = VisualManager(vis, 'contestai')


#%%
for epoch in range(opt['epochs']):
    for batch in training_generator_ok:
        # training
        print('Training code here')




