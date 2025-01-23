from utils import extract_feature_video, load_data, mapping_action_name_to_label
from build_model import LSTMModel
# from torchsummary import summary
from torchinfo import summary
import time
import torch

PATH_VIDEO_TEST = "actual_test/test_3.mp4"

data = extract_feature_video(PATH_VIDEO_TEST)

model = LSTMModel(256, 1000, 2, 101)
model.load_state_dict(torch.load("trained_model_v2.pth"))
model.eval()
model.to("cuda")

data = data.unsqueeze(0)
predict = model(data.to("cuda"))

print(torch.argmax(predict))

# INFOR_DATASET_PATH = "dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt"

# print(len(mapping_action_name_to_label(INFOR_DATASET_PATH)))

# start_time = time.time()
# video_path = 'dataset/UCF101/UCF-101/BaseballPitch/v_BaseballPitch_g01_c01.avi'

# print(extract_feature_video(video_path).shape)

# print(time.time() - start_time)

# #torch.Size([100, 1000])

# model = LSTMModel(256, 1000, 2, 101)
# model.to("cuda")

# summary(model, input_size=(128, 100, 1000))

# # ==========================================================================================
# # Layer (type:depth-idx)                   Output Shape              Param #
# # ==========================================================================================
# # LSTMModel                                [128, 101]                --
# # ├─LSTM: 1-1                              [128, 100, 256]           1,814,528
# # ├─Linear: 1-2                            [128, 101]                25,957
# # ==========================================================================================
# # Total params: 1,840,485
# # Trainable params: 1,840,485
# # Non-trainable params: 0
# # Total mult-adds (Units.GIGABYTES): 23.23
# # ==========================================================================================
# # Input size (MB): 51.20
# # Forward/backward pass size (MB): 26.32
# # Params size (MB): 7.36
# # Estimated Total Size (MB): 84.88
# # ==========================================================================================