from torch.utils.data import Dataset

class UCFDataset(Dataset):
    def __init__(self, data, mapping, list_video_name):
        super().__init__()

        self.data = data
        self.mapping = mapping
        self.list_video_name = list_video_name

    def __len__(self):
        return len(self.list_video_name)
    
    def __getitem__(self, index):
        video_name = self.list_video_name[index]
        action_name, _ = video_name.split("/")
        return self.data[video_name], self.mapping[action_name]
    