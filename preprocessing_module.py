import os
from torch.utils.data import Dataset
from skimage import io


class BeanTechDataset(Dataset):
    """BeanTech dataset."""

    def __init__(self, images_arrays_list, images_labels_list, images_names_list, transform=None):
        """
        Args:
            images_arrays_list (list): List containing all images represented as numpy arrays.
            images_labels_list (list): List containing all images labels.
            images_names_list (list): List containing all images names.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_arrays_list = images_arrays_list
        self.images_labels_list = images_labels_list
        self.images_names_list = images_names_list
        self.transform = transform

    def __len__(self):
        return len(self.images_names_list)

    def __getitem__(self, idx):
        image = self.images_arrays_list[idx]
        image_label = self.images_labels_list[idx]
        image_name = self.images_names_list[idx]
        sample = {'image': image, 'image_label': image_label, 'image_name': image_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def process_original_data(path_to_folder, dataset_name, images_type: float, transform):
    """
    Processes the original data and returns the same data organized in one or more BeanTechDataset object(s):\n
    - one BeanTechDataset object containing all the data if dataset_name = 'all'
    - one BeanTechDataset object containing only albedo data if dataset_name = 'albedo'
    - one BeanTechDataset object containing only raw data if dataset_name = 'nsew'
    - two BeanTechDataset objects (one for 'Albedo' images, one for all other images) if dataset_name = 'albedo_nsew';
    - five BeanTechDataset objects (one for 'Albedo' images, one for 'East' images, one for 'North' images, one for
      'South' images, one for 'West' images) if dataset_name = 'albedo_nsew_split'.

    :param path_to_folder: path to the folder containing the images
    :type path_to_folder: string
    :param dataset_name: name of the dataset
    :type dataset_name: string
    :param images_type: type of images ('OK' or 'KO')
    :type images_type: float
    :return: one, two or five BeanTechDataset objects, depending on the value of number_of_desired_datasets
    """
    # create empty lists
    names_all, names_albedo, names_other, names_east, names_north, names_south, names_west = ([] for i in range(7))
    arrays_all, arrays_albedo, arrays_other, arrays_east, arrays_north, arrays_south, arrays_west = ([] for i in range(7))
    # list of files in the folder
    files_list = os.listdir(path_to_folder)
    # sort the list alphabetical order
    files_list.sort()
    # reverse the list (pop() method removes and returns the last item)
    files_list.reverse()

    def load_image(direction_names_list, direction_images_list):
        if dataset_name == 'all':
            names_all.append(file_name)
            arrays_all.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'albedo_nsew' or dataset_name == 'nsew':
            names_other.append(file_name)
            arrays_other.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'albedo_nsew_split':
            direction_names_list.append(file_name)
            direction_images_list.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'albedo':
            return

    # process the files
    while files_list:
        # albedo
        file_name = files_list.pop()
        numeric_file_id = file_name.split('_')[0]
        if dataset_name == 'all':
            names_all.append(file_name)
            arrays_all.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'nsew':
            break
        else:
            names_albedo.append(file_name)
            arrays_albedo.append(io.imread(path_to_folder + '/' + file_name)/255.0)

        # east (same numeric id)
        file_name = numeric_file_id + '_East.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_east, arrays_east)
        # north (same numeric id)
        file_name = numeric_file_id + '_North.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_north, arrays_north)
        # south (same numeric id)
        file_name = numeric_file_id + '_South.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_south, arrays_south)
        # west (same numeric id)
        file_name = numeric_file_id + '_West.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_west, arrays_west)

    # create datasets
    if dataset_name == 'all':
        labels = [float(images_type) for i in range(len(names_all))]
        dataset_all = BeanTechDataset(arrays_all, labels, names_all)
        return dataset_all
    elif dataset_name == 'albedo_nsew':
        labels = [float(images_type) for i in range(len(names_albedo))]
        dataset_albedo = BeanTechDataset(arrays_albedo, labels, names_albedo, transform=transform)
        labels = [float(images_type) for i in range(len(names_other))]
        dataset_other = BeanTechDataset(arrays_other, labels, names_other, transform=transform)
        return dataset_albedo, dataset_other
    elif dataset_name == 'albedo_nsew_split':
        labels = [float(images_type) for i in range(len(names_albedo))]
        dataset_albedo = BeanTechDataset(arrays_albedo, labels, names_albedo, transform=transform)
        dataset_east = BeanTechDataset(arrays_east, labels, names_east, transform=transform)
        dataset_north = BeanTechDataset(arrays_north, labels, names_north, transform=transform)
        dataset_south = BeanTechDataset(arrays_south, labels, names_south, transform=transform)
        dataset_west = BeanTechDataset(arrays_west, labels, names_west, transform=transform)
        return dataset_albedo, dataset_east, dataset_north, dataset_south, dataset_west
    elif dataset_name == 'albedo':
        labels = [float(images_type) for i in range(len(names_albedo))]
        dataset_albedo = BeanTechDataset(arrays_albedo, labels, names_albedo, transform=transform)
        return dataset_albedo
    elif dataset_name == 'nsew':
        labels = [float(images_type) for i in range(len(names_other))]
        dataset_other = BeanTechDataset(arrays_other, labels, names_other, transform=transform)
        return dataset_other

