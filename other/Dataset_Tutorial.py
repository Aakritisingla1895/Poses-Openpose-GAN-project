import torch
from torch.utils.data import Dataset, DataLoader

# create to random tensors representing dummy data for the dataset
data_count = 100 # number of data instances
data_dim = 8 # number of features per instance
label_count = 4 # number of class labels

dummy_features = torch.rand([data_count, data_dim], dtype=torch.float32)
dummy_labels = torch.randint(0, label_count, [data_count], dtype=torch.int32)

# Create class for a simple customised Dataset by subclassing Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# Create an instance of the customised dataset
customDataset = CustomDataset(dummy_features, dummy_labels)

#print length of dataset
print(len(customDataset ))

#iterate over all data instances contained in dataset
for instance in iter(customDataset):
    print(instance)
    
# split full dataset into train and test dataset
train_test_ratio = 0.8 # 80% of data goes into training set, 20% into test set
train_size = int(len(customDataset) * train_test_ratio)
test_size = len(customDataset) - train_size 

train_dataset, test_dataset = torch.utils.data.random_split(customDataset , [train_size, test_size])

print("train dataset size: ", len(train_dataset))
print("test dataset size: ", len(test_dataset))

# Instantiate Dataloaders

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

batch = next(iter(train_dataloader))

batch_features = batch[0]
batch_labels = batch[1]

print("batch features shape ", batch_features.shape)
print("batch labels shape", batch_labels.shape)

# iterate over DataLoader for test dataset

for (idx, batch) in enumerate(test_dataloader):
    print("batch ", idx, " features: ", batch[0])
    print("batch ", idx, " labels: ", batch[1])