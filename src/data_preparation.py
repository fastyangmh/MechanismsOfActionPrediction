#import
from src.project_parameters import ProjectParameters
from torch.utils.data import Dataset
from typing import TypeVar
T_co = TypeVar('T_co', covariant=True)


#class
class MyMoADataset(Dataset):
    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        sample = self.data[index]
        target = self.targets[index]
        return sample, target


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    #
    import pandas as pd
    from os.path import join
    data = pd.read_csv(join(project_parameters.root, 'train_features.csv'))
    columns = data.columns.values[1:]
    data = data.loc[:, columns].values
    label = pd.read_csv(
        join(project_parameters.root, 'train_targets_scored.csv'))
    columns = label.columns.values[1:]
    label = label.loc[:, columns].values

    dataset = MyMoADataset(data=data, targets=label)
