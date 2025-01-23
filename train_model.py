import torch.nn as nn
from build_model import LSTMModel
from torch.utils.data import DataLoader
from utils import load_txt_file, train, load_data, mapping_action_name_to_label
from UCFDataset import UCFDataset
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    INPUT_DIM = 1000
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_CLASSES = 101
    MAX_LEN = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    INFOR_DATASET_PATH = "dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt"
    DATASET_PATH = "dataset/UCF101/UCF-101/"

    # infor_data_train, infor_data_test = load_txt_file(INFOR_DATASET_PATH)
    infor_data_train, list_video_name_train = load_data("train_set.pth")
    infor_data_test, list_video_name_test = load_data("test_set.pth")

    mapping = mapping_action_name_to_label(INFOR_DATASET_PATH)

    train_dataset = UCFDataset(infor_data_train, mapping, list_video_name_train)
    test_dataset = UCFDataset(infor_data_test, mapping, list_video_name_test)

    print(f"Training size: {len(train_dataset)}, Testing size: {len(test_dataset)}")

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM, num_classes=NUM_CLASSES, num_layers=NUM_LAYERS)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    torch.save(model.state_dict(), "trained_model_v2.pth")
    print("Model saved successfully.")