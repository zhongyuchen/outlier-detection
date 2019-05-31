from dataset import get_dataset
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import OneClassSVM
import time


def plot_outlier(data, outliers, rbf, nu):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	print(data[10])
	ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b', label='normal')
	ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='r', label='outliers')
	ax.set_xlabel('L')
	ax.set_ylabel('F')
	ax.set_zlabel('M')
	ax.legend()
	plt.title('kernel=%s, nu=%.2f' % (rbf, nu))
	plt.show()


def svdd(data, rbf, nu):
	# normalize
	for i in range(3):
		data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
	to_index = np.arange(len(data))

	# svm
	ocsvm = OneClassSVM(kernel=kernel, gamma='auto', tol=1e-3, nu=nu, shrinking=True, max_iter=-1)
	ocsvm.fit(data)
	pred = ocsvm.predict(data)
	normal = to_index[pred == 1]
	outliers = to_index[pred == -1]

	return data[normal], data[outliers]


if __name__ == "__main__":
	data = get_dataset()
	kernel = 'linear'
	nu = 0.1
	start_time = time.time()
	normal, outliers = svdd(data, rbf=kernel, nu=nu)
	print('Time used:', time.time() - start_time)
	print('outlier:', len(outliers))
	plot_outlier(normal, outliers=outliers, rbf=kernel, nu=nu)
