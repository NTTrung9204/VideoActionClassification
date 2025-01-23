from utils import load_txt_file, save_data

INFOR_DATASET_PATH = "dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt"
DATASET_PATH = "dataset/UCF101/UCF-101/"

infor_data_train, infor_data_test = load_txt_file(INFOR_DATASET_PATH)

save_data(infor_data_test, DATASET_PATH, "test_set")

save_data(infor_data_train, DATASET_PATH, "train_set")
