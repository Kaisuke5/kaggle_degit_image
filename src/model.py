import numpy as np
import six
import chainer.links as L
import chainer
print chainer.__version__
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import pickle

class network(chainer.Chain):

	def __init__(self, n_in, n_units, n_out,):
		super(network, self).__init__(
			l1=L.Linear(n_in, n_units),
			l2=L.Linear(n_units, n_units),
			l3=L.Linear(n_units, n_out),
		)

	def __call__(self, x, t):
		y = self.predict(x)
		self.loss = F.softmax_cross_entropy(y, t)
		self.accuracy = F.accuracy(y, t)
		return self.loss

	def predict(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		t = self.l3(h2)
		return t


class degit_network():

	def __init__(self,units=300,gpu=-1):
		self.units = units
		self.model = None
		self.gpu = gpu

		self.gpu = gpu
		if self.gpu >= 0:
			from chainer import cuda
			self.xp = cuda.cupy
			cuda.get_device(self.gpu).use()
		else:
			self.xp = np


	def fit(self,x_train,y_train,n_epoch=100, batchsize=300,save=False):
		x_train = np.array(x_train, np.float32)
		y_train = np.array(y_train, np.int32)
		self.train(x_train, y_train, n_epoch=n_epoch, batchsize=batchsize)

		if save:
			file_name = 'model' + '_u' + str(self.units) + '_n' + str(n_epoch) + '_b' + str(batchsize)+'.pkl'
			with open(file_name, mode='wb') as f:
				pickle.dump(self.model, f, -1)


	def train(self, x_train, y_train,n_epoch=100, batchsize=30):
		print 'n_epoch:%d,batch_size:%d,units:%d' %(n_epoch,batchsize,self.units)
		self.model = network(len(x_train[0]),self.units,10)
		if self.gpu >= 0:
			print 'model to gpu'
			self.model.to_gpu()
			print type(self.model)




		optimizer = optimizers.Adam()
		optimizer.setup(self.model)
		N = len(x_train)
		for epoch in six.moves.range(1, n_epoch + 1):
			print('epoch', epoch)
			# training
			perm = np.random.permutation(N)
			sum_loss = 0
			accuracy = 0
			for i in six.moves.range(0, N, batchsize):

				if self.gpu >= 0:
					x = chainer.Variable(cuda.to_gpu(x_train[perm[i:i + batchsize]]))
					t = chainer.Variable(cuda.to_gpu(y_train[perm[i:i + batchsize]]))
				else:
					x = chainer.Variable(x_train[perm[i:i + batchsize]])
					t = chainer.Variable(y_train[perm[i:i + batchsize]])

				optimizer.update(self.model, x, t)
				sum_loss += float(self.model.loss.data) * len(t.data)
				accuracy += float(self.model.accuracy.data) * len(t.data)

			print 'epoch %d mean_squared_error:%f accuracy:%f' % (epoch,sum_loss/N,accuracy/N)



	# def to_gpu(self, device=None):
	# 	with cuda.get_device(device):
	# 		super(chainer.Chain, self).to_gpu()
	# 		d = self.__dict__
	# 		for name in self._children:
	# 			d[name].to_gpu()
	# 	return self


	def predict(self,test_x):
		x = chainer.Variable(self.xp.asarray(test_x, self.xp.float32))
		result = self.model.predict(x).data
		if self.gpu >=0:
			result = cuda.to_cpu(result)

		return result.argmax(axis=1)







