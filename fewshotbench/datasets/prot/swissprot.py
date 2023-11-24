from abc import ABC
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from torch.utils.data import DataLoader

from datasets.dataset import *
from datasets.prot.utils import *

EMB_PATH = 'embeddings'
EMB_LAYER = 33
PROTDIM = 1280

@dataclass_json
@dataclass
class ProtSample:
    input_seq: torch.Tensor
    annot: torch.Tensor
    entry: str


class SPDataset(FewShotDataset, ABC):
    _dataset_name = 'swissprot'
    _dataset_url = 'https://drive.google.com/u/0/uc?id=1a3IFmUMUXBH8trx_VWKZEGteRiotOkZS&export=download'

    def load_swissprot(self, level = 5, mode='train', min_samples =20):
        # samples = get_samples(root = self.data_dir, level=level)
        samples = get_samples_using_ic(root = self.data_dir)
        samples = check_min_samples(samples, min_samples)

        unique_ids = set(get_mode_ids(samples)[mode])

        return [sample for sample in samples if sample.annot in unique_ids]
        

class SPSimpleDataset(SPDataset):
    def __init__(self, batch_size, root='./data/', mode='train', min_samples=20):
        self.initialize_data_dir(root, download_flag=False)
        self.samples = self.load_swissprot(mode = mode, min_samples = min_samples)
        self.batch_size = batch_size
        self.encoder = encodings(self.data_dir)
        super().__init__()

    def __getitem__(self, i):
        sample = self.samples[i]
        return sample.input_seq, self.encoder[sample.annot]

    def __len__(self):
        return len(self.samples)

    @property
    def dim(self):
        return self.samples[0].input_seq.shape[0]

    def get_data_loader(self) -> DataLoader:
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader


class SPSetDataset(SPDataset):

    def __init__(self, n_way, n_support, n_query, n_episode=100, root='./data', mode='train'):
        self.initialize_data_dir(root, download_flag=False)

        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query
        self.encoder = encodings(self.data_dir)

        samples_all= self.load_swissprot(mode = mode, min_samples = min_samples)


        self.categories = get_ids(samples_all) # Unique annotations
        self.x_dim = PROTDIM

        self.sub_dataloader = []

        sub_data_loader_params = dict(batch_size=min_samples,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for annotation in self.categories:
            samples = [sample for sample in samples_all if sample.annot == annotation]
            sub_dataset = SubDataset(samples, self.data_dir)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        super().__init__()

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.categories)

    @property
    def dim(self):
        return self.x_dim

    def get_data_loader(self) -> DataLoader:
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader

class SubDataset(Dataset):
    def __init__(self, samples, data_dir):
        self.samples = samples
        self.encoder = encodings(data_dir)

    def __getitem__(self, i):
        sample = self.samples[i]
        return sample.input_seq, self.encoder[sample.annot]
        

    def __len__(self):
        return len(self.samples)

    @property
    def dim(self):
        return PROTDIM

if __name__ == "__main__":
    d = SPSetDataset(5, 5, 15)
