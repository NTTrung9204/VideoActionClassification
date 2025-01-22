from torch.utils.data import Dataset
from utils import extract_feature_video

class UCFDataset(Dataset):
    def __init__(self, data, folder_path):
        super().__init__()

        self.data = data
        self.folder_path = folder_path

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        video_name, label = self.data[index].split()
        return extract_feature_video(self.folder_path + video_name), label
    