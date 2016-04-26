import pickle
import pandas as pd
TRAIN_PATH = '../data/train.csv'
TEST_PATH = '../data/test.csv'

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

train_y = train_data['label'].values
del train_data['label']
train_x = train_data.values
test_x = test_data.values

data = [train_x, train_y, test_x]

with open('data.pkl','wb') as f:
	pickle.dump(data,f,-1)
