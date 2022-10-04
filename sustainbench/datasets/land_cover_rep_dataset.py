from sustainbench.datasets.sustainbench_dataset import SustainBenchDataset

import os
import pandas as pd
import torch
import torchvision.transforms as T
import numpy as np

IMG_DIM = 100
NUM_CLASSES = 61
BANDS = 4

class LandCoverRepDataset(SustainBenchDataset):
    """
    The Land Cover Representation dataset.
    Imagery is from the USDA's National Agriculture Imagery Program (NAIP), which provides aerial imagery for public
    use that has four spectral bands (red (R), green (G), blue (B), and infrared (N)) at 0.6 m ground resolution.
    The output is image-level land cover classification (61 possible classes), where labels are generated from a 
    high-quality USDA dataset [1]. Since our 100x100 px NAIP tiles and the USDA land cover dataset are not perfectly 
    aligned, the land cover class used for the label is the mode across pixels in the tile.
    This dataset contains 100k tile triplets, with each triplet identified by its index and "anchor", "neighbor", or 
    "distant". However, for the autoencoding task, triplets will be separated, meaning 300k tiles in total.
    Input (x):
        5 x 100 x 100 RGBN (red (R), green (G), blue (B), infrared (N), land cover segmentation mask) satellite image.
    Label (y):
        y is one of 61 land cover classes
    Website: https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg15/land_cover_representation.html
    Original publication:
    @article{jean2019tile2vec,
        Author = {Jean, Neal and Wang, Sherrie and Samar, Anshul and Azzari, George and Lobell, David and Ermon, Stefano},
        Journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
        Month = {Jul.},
        Number = {01},
        Pages = {3967-3974},
        Title = {Tile2Vec: Unsupervised Representation Learning for Spatially Distributed Data},
        Volume = {33},
        Year = {2019}}
    References:
        [1] National Agricultural Statistics Service. USDA National Agricultural Statistics Service Cropland Data 
        Layer. Published crop-specific data layer [Online], 2018. URL https://nassgeodata.594gmu.edu/CropScape/.
    License:
        Distributed under the SustainBench MIT License.
        https://github.com/sustainlab-group/sustainbench/blob/main/LICENSE
    """
    _dataset_name = 'land_cover_rep'
    _versions_dict = {
        '1.0': {'download_url': None},
        '1.1': {'download_url': None}
    }

    def __init__(self, root_dir, version="1.1", split="train", **dataset_kwargs):
        self._version = version
        root = self.root = os.path.expanduser(root_dir)
        self._data_dir = self.initialize_data_dir(root, download=False)
        self._tile_dir = os.path.join(self._data_dir, 'tile_triplets')
        self._metadata_fields = ['fileName', 'landCover', 'split', 'splitStr']
        self._metadata_array = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        self._metadata_map = \
            {'landCover': {
                1: 'Corn',
                2: 'Cotton',
                3: 'Rice',
                4: 'Sorghum',
                12: 'Sweet Corn',
                21: 'Barley',
                22: 'Durum Wheat',
                23: 'Spring Wheat',
                24: 'Winter Wheat',
                27: 'Rye',
                28: 'Oats',
                33: 'Safflower',
                36: 'Alfalfa',
                37: 'Other Hay/Non Alfalfa',
                42: 'Dry Beans',
                44: 'Other Crops',
                48: 'Watermelons',
                49: 'Onions',
                53: 'Peas',
                54: 'Tomatoes',
                59: 'Sod/Grass Seed',
                61: 'Fallow/Idle Cropland',
                66: 'Cherries',
                67: 'Peaches',
                69: 'Grapes',
                71: 'Other Tree Crops',
                72: 'Citrus',
                74: 'Pecans',
                75: 'Almonds',
                76: 'Walnuts',
                77: 'Pears',
                111: 'Open Water',
                121: 'Developed/Open Space',
                122: 'Developed/Low Intensity',
                123: 'Developed/Med Intensity',
                124: 'Developed/High Intensity',
                131: 'Barren',
                141: 'Deciduous Forest',
                142: 'Evergreen Forest',
                152: 'Shrubland',
                176: 'Grassland/Pasture',
                190: 'Woody Wetlands',
                195: 'Herbaceous Wetlands',
                204: 'Pistachios',
                205: 'Triticale',
                206: 'Carrots',
                207: 'Asparagus',
                208: 'Garlic',
                209: 'Cantaloupes',
                211: 'Olives',
                212: 'Oranges',
                213: 'Honeydew Melons',
                217: 'Pomegranates',
                218: 'Nectarines',
                220: 'Plums',
                225: 'Dbl Crop WinWht/Corn',
                226: 'Dbl Crop Oats/Corn',
                227: 'Leccuce',
                236: 'Dbl Crop WinWht/Sorghum',
                237: 'Dbl Crop Barley/Corn',
                242: 'Blueberries'
            }}
        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
        self._split_array = self._metadata_array['split'].values
        self._filenames = self._metadata_array['fileName'].values
        # y_array stores idx ids corresponding to land cover class.
        self._y_array = torch.from_numpy(self._metadata_array['landCover'].values)
        self._y_size = 1
        self._n_classes = NUM_CLASSES
        self._original_resolution = IMG_DIM
        self.check_init()
        
    def __len__(self):
        return len(self.y_array)
    
    def __getitem__(self, idx):
        # Any transformations are handled by the LandCoverRepSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        return x, y

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        data = np.load(os.path.join(self._tile_dir, self._filenames[idx]))
        return data

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (LandCoverRepSubset): A (potentially subsampled) subset of the LandCoverRepDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = LandCoverRepSubset(self, split_idx, transform)
        return subset

    def check_init(self):
        """
        Convenience function to check that the SustainBenchDataset is properly configured.
        """
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'LandCoverRepDataset is missing {attr_name}.'
        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')
        # Check splits
        assert self.split_dict.keys()==self.split_names.keys()
        assert 'train' in self.split_dict
        assert 'val' in self.split_dict
        assert isinstance(self.y_array, torch.Tensor), 'y_array must be a torch.Tensor'
        # Check that dimensions match
        assert len(self.y_array) == len(self.metadata_array)
        assert len(self.y_array.shape) == self.y_size
        assert len(self.split_array) == len(self.metadata_array)
        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]
   
    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'amazon', 'camelyon17'.
        """
        return self._dataset_name

    @property
    def data_dir(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_dir

    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)

    @property
    def split_scheme(self):
        """
        A string identifier of how the split is constructed,
        e.g., 'standard', 'in-dist', 'user', etc.
        """
        return self._split_scheme

    @property
    def split_dict(self):
        """
        A dictionary mapping splits to integer identifiers (used in split_array),
        e.g., {'train': 0, 'val': 1, 'test': 2}.
        Keys should match up with split_names.
        """
        return getattr(self, '_split_dict', None)

    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        e.g., {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.
        Keys should match up with split_dict.
        """
        return getattr(self, '_split_names', None)

    @property
    def split_array(self):
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def y_size(self):
        """
        The number of dimensions/elements in the target, i.e., len(y_array[i]).
        For standard classification/regression tasks, y_size = 1.
        For multi-task or structured prediction settings, y_size > 1.
        Used for logging and to configure models to produce appropriately-sized output.
        """
        return self._y_size

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def is_classification(self):
        """
        Boolean. True if the task is classification, and false otherwise.
        Used for logging purposes.
        """
        return (self.n_classes is not None)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'splitStr'].
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Pandas DataFrame of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

    @property
    def metadata_map(self):
        """
        An optional dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['landCover', 'y']
            metadata_map = {'landCover': {61: 'Fallow/Idle Cropland', 66: 'Cherries'}}
        then if metadata_array['landCover'][i] == 61, the i-th data point belongs to the 'Fallow/Idle Cropland' class
        while if metadata_array['landCover'][i] == 66, it belongs to the 'Cherries' class.
        """
        return getattr(self, '_metadata_map', None)

    @property
    def original_resolution(self):
        """
        Original image resolution for image datasets.
        """
        return getattr(self, '_original_resolution', None)

class LandCoverRepSubset(LandCoverRepDataset):
    def __init__(self, dataset, indices, transform):
        """
        This acts like torch.utils.data.Subset.
        We pass in transform explicitly because it can potentially vary at
        training vs. test time, if we're using data augmentation.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
    
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.indices)
    
    @property
    def split_array(self):
        return self.dataset.split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset.y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

"""
Custom torchvision transforms
"""
class RandomMask(object):
    """
    Randomly drop out parts of the input image. All channels are included.
    """
    def __init__(self, p=0.5):
        assert isinstance(p, float)
        assert p >= 0 and p <= 1, "probability p must be between 0 and 1"
        self.p = p

    def __call__(self, x):
        _, h, w = x.shape
        mask = (torch.rand(1, h, w)+(1-self.p)).floor()
        masked_x = mask*x
        return masked_x

class GetBands(object):
    """
    Gets the first X bands of the tile.
    """
    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands
    def __call__(self, x):
        # Tiles are already in [c, w, h] order
        return x[:self.bands,:,:]

"""
Helper functions for retrieving datasets
"""
def get_train_val_datasets(dataset_name, dataset_root, train_size, validation_size):
    if dataset_name == 'LandCoverRep':
        transforms = T.Compose([T.ToTensor(), GetBands(3), T.RandomCrop(train_size),
                                T.RandomHorizontalFlip(), T.RandomVerticalFlip()])
        dataset = LandCoverRepDataset(root=dataset_root)
        train_dataset = dataset.get_subset('train', transform=transforms)
        validation_dataset = \
            dataset.get_subset('val', transform=T.Compose([T.ToTensor(), GetBands(3), T.CenterCrop(validation_size)]))
    else:
        raise RuntimeError(
            'Dataset not supported.'
            'Pick \'LandCoverRep\'.'
        )
    return train_dataset, validation_dataset

def get_inf_dataset(dataset_name, dataset_root, inference_size):
    if dataset_name == 'LandCoverRep':
        transforms = T.Compose([T.ToTensor(), GetBands(3), T.CenterCrop(inference_size)])
        dataset = LandCoverRepDataset(root=dataset_root)
        inf_dataset = dataset.get_subset('test', transform=transforms)
    else:
        raise RuntimeError(
            'Dataset not supported.'
            'Pick \'LandCoverRep\'.'
        )
    return inf_dataset
