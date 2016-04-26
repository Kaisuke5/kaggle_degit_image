import model
import pandas as pd
import numpy as np
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import model
import argparse

parser = argparse.ArgumentParser(description='kaggle kobe')
parser.add_argument('--gpu', '-g', default=-1, type=int,help='gpu -1')
args = parser.parse_args()

TRAIN_PATH = '../data/train.csv'
TEST_PATH = '../data/test.csv'

print 'loading data'
data = pickle.load(open('data.pkl','rb'))

train_x = np.asarray(data[0], np.int32)
train_y = np.asarray(data[1], np.int32)
test_x = np.asarray(data[2], np.float32)


print train_x[0].dtype,train_y.dtype

NN = model.degit_network(units=500,gpu=args.gpu)
NN.fit(train_x,train_y,n_epoch=100,batchsize=200)
result = NN.predict(test_x)
print len(result)

f = open('output.csv','w')
print 'writing answear'
for r in result:
	f.write(str(r)+'\n')

f.close()