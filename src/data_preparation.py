#import
from src.project_parameters import ProjectParameters
from torch.utils.data import Dataset
from DeepLearningTemplate.data_preparation import BaseLightningDataModule
import numpy as np
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
import random
from typing import TypeVar, Optional, Callable
T_co = TypeVar('T_co', covariant=True)


#def
def create_datamodule(project_parameters):
    return MoALightningDataModule(
        root=project_parameters.root,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config)


#class
class MyMoADataset(Dataset):
    def __init__(
        self,
        data,
        targets,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.cp_type = {'ctl_vehicle': 0, 'trt_cp': 1}
        self.cp_dose = {'D1': 0, 'D2': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        sample = self.data[index]
        idx = [1] + list(range(3, len(sample)))
        onehot = []
        for v in sample[[0, 2]]:
            if v in self.cp_type:
                onehot.append(np.eye(len(self.cp_type))[self.cp_type[v]])
            elif v in self.cp_dose:
                onehot.append(np.eye(len(self.cp_dose))[self.cp_dose[v]])
        onehot = np.concatenate(onehot)
        sample = np.append(sample[idx], onehot).astype(np.float32)
        sample = sample[None]
        target = self.targets[index].astype(np.float32)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class MoALightningDataModule(BaseLightningDataModule):
    def __init__(self, root, classes, max_samples, batch_size, num_workers,
                 device, transforms_config, target_transforms_config):
        super().__init__(root=root,
                         predefined_dataset=None,
                         classes=classes,
                         max_samples=max_samples,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         device=device,
                         transforms_config=transforms_config,
                         target_transforms_config=target_transforms_config)

    def prepare_data(self) -> None:
        # load the features and targets from csv
        data = pd.read_csv(join(self.root, 'train_features.csv'))
        columns = data.columns.values[1:]
        data = data.loc[:, columns].values
        targets = pd.read_csv(join(self.root, 'train_targets_scored.csv'))
        columns = targets.columns.values[1:]
        targets = targets.loc[:, columns].values
        self.data = data
        self.targets = targets

    def setup(self, stage: Optional[str] = None) -> None:
        if self.max_samples is not None:
            index = random.sample(population=range(len(self.data)),
                                  k=self.max_samples)
            self.data = self.data[index]
            self.targets = self.targets[index]
        x_train, x_val, y_train, y_val = train_test_split(
            self.data, self.targets, test_size=self.val_size)
        self.train_dataset = MyMoADataset(
            data=x_train,
            targets=y_train,
            transform=self.transforms_dict['train'],
            target_transform=self.target_transforms_dict['train'])
        self.val_dataset = MyMoADataset(
            data=x_val,
            targets=y_val,
            transform=self.transforms_dict['val'],
            target_transform=self.target_transforms_dict['val'])
        self.test_dataset = MyMoADataset(
            data=x_val,
            targets=y_val,
            transform=self.transforms_dict['test'],
            target_transform=self.target_transforms_dict['test'])


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print(
        'the dimension of target: {}'.format(1 if type(y) == int else y.shape))
