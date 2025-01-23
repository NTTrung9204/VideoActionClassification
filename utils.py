import imageio
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import random
import torch
from sklearn.metrics import accuracy_score
import sys

import warnings

warnings.filterwarnings('ignore')

FRAMES = 50

extracted_model = models.inception_v3(pretrained=True)
extracted_model.to("cuda")
extracted_model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def extract_feature_video(video_path):
    reader = imageio.get_reader(video_path)

    total_frames = reader.count_frames()

    frame_step = total_frames / FRAMES

    frame_step = max(1, frame_step)

    extracted_images = []

    for i in range(FRAMES):
        frame_index = min(int(i * frame_step), total_frames - 1)
        frame = reader.get_data(frame_index)

        pil_image = Image.fromarray(frame)

        transform_frame = transform(pil_image)
        extracted_images.append(transform_frame)

    reader.close()

    extracted_images_np = np.array(extracted_images)

    extracted_images_torch = torch.tensor(extracted_images_np, dtype=torch.float32).to("cuda")

    with torch.no_grad():
        return extracted_model(extracted_images_torch)
    
def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    train_size = int(0.8 * len(lines))

    random.shuffle(lines)

    train_data = lines[:train_size]
    test_data = lines[train_size:]

    return train_data, test_data

def save_data(infor_data, DATASET_PATH, saved_name):
    dict = {}
    data_size = len(infor_data)
    for index, data in enumerate(infor_data):
        sys.stdout.write(f"\rExtracting [{index:4d}|{data_size}]")

        video_name, label = data.split()

        dict[video_name] = extract_feature_video(DATASET_PATH + video_name)

    torch.save(dict, f"{saved_name}.pth")

    print("Save successfully!")

def load_data(decode_data):
    encode_data = torch.load(decode_data)

    return encode_data, list(encode_data.keys())

def mapping_action_name_to_label(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    mapping = {}

    for line in lines:
        video_name, label = line.split()

        action_name, _ = video_name.split("/")

        mapping[action_name] = int(label) - 1

    return mapping

def train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        valid_accuracy = evaluate(model, valid_loader, device)
        valid_accuracies.append(valid_accuracy)

        print(f"Epoch [{epoch + 1:4d}/{num_epochs}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

    return train_losses, valid_accuracies


def evaluate(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Predicted class labels
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy