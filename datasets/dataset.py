import os
import time
from abc import abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive


# Inspired/modified from WILDS (https://wilds.stanford.edu/)
# and COMET (https://github.com/snap-stanford/comet)

class FewShotDataset(Dataset):

    def __init__(self):
        self.check_init()

    def check_init(self):
        """
        Convenience function to check that the FewShotDataset is properly configured.
        """
        required_attrs = ['_dataset_name', '_data_dir']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(f'FewShotDataset must have attribute {attr}.')

        if not os.path.exists(self._data_dir):
            raise ValueError(
                f'{self._data_dir} does not exist yet. Please generate/download the dataset first.')


    @abstractmethod
    def __getitem__(self, i):
        return NotImplemented

    @abstractmethod
    def __len__(self):
        return NotImplemented

    @property
    @abstractmethod
    def dim(self):
        return NotImplemented

    @abstractmethod
    def get_data_loader(self, mode='train') -> DataLoader:
        return NotImplemented

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'tabula_muris'
        """
        return self._dataset_name

    @property
    def data_dir(self):
        return self._data_dir

    def initialize_data_dir(self, root_dir, download_flag=True):
        os.makedirs(root_dir, exist_ok=True)
        self._data_dir = os.path.join(root_dir, self._dataset_name)
        if not self.dataset_exists_locally():
            if not download_flag:
                raise FileNotFoundError(
                    f'The {self._dataset_name} dataset could not be found in {self._data_dir}. Please'
                    f' download manually. '
                )

            self.download_dataset(download_flag)

    def download_dataset(self, download_flag):
        if self._dataset_url is None:
            raise ValueError(f'{self._dataset_name} cannot be automatically downloaded. Please download it manually.')

        print(f'Downloading dataset to {self._data_dir}...')

        try:
            start_time = time.time()
            download_and_extract_archive(
                url=self._dataset_url,
                download_root=self._data_dir,
                remove_finished=True)
            download_time_in_minutes = (time.time() - start_time) / 60
            print(f"\nIt took {round(download_time_in_minutes, 2)} minutes to download and uncompress the dataset.\n")
        except Exception as e:
            print(f"Exception: ", e)

    def dataset_exists_locally(self):
        return os.path.exists(self._data_dir) and (len(os.listdir(self._data_dir)) > 0 or (self._dataset_url is None))


class FewShotSubDataset(Dataset):
    def __init__(self, samples, category):
        self.samples = samples
        self.category = category

    def __getitem__(self, i):
        return self.samples[i], self.category

    def __len__(self):
        return self.samples.shape[0]

    @property
    def dim(self):
        return self.samples.shape[1]


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
