from torch.utils.data import Dataset

class TestDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return idx