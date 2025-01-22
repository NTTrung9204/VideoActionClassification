import torch.nn as nn
from build_model import LSTMModel
from torch.utils.data import DataLoader
from utils import load_txt_file, train
from UCFDataset import UCFDataset
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    INPUT_DIM = 1000
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_CLASSES = 101
    MAX_LEN = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 1
    INFOR_DATASET_PATH = "dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt"
    DATASET_PATH = "dataset/UCF101/UCF-101/"

    infor_data_train, infor_data_test = load_txt_file(INFOR_DATASET_PATH)

    train_dataset = UCFDataset(infor_data_train, DATASET_PATH)
    test_dataset = UCFDataset(infor_data_test, DATASET_PATH)

    print(f"Training size: {len(train_dataset)}, Testing size: {len(test_dataset)}")

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for batch in train_data_loader:
        data, label = batch
        print(data.shape)
        print(label)
        
        exit()

    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = LSTMModel(hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM, num_classes=NUM_LAYERS, num_layers=NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses, valid_accuracies = train(model, train_data_loader, test_data_loader, criterion, optimizer, device, NUM_EPOCHS)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.show()

    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved successfully.")