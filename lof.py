from dataset import get_dataset
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import time

data = get_dataset()
kdn = []  # list of k neighbor index
dis = []  # list of k neighbor distance
lrd = np.zeros(len(data))  # list of local reachability density
lof = np.zeros(len(data))  # list of lof


def plot_outlier(data, outliers=None, k=None, threshold=None):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b', label='normal')
	if outliers is not None:
		ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='r', label='outliers')
	ax.set_xlabel('L')
	ax.set_ylabel('F')
	ax.set_zlabel('M')
	ax.legend()
	if k is not None and threshold is not None:
		plt.title('k=%d, lof>=%.2f' % (k, threshold))
	plt.show()


def distance(p):
	return np.sqrt(np.sum((p - data) ** 2, axis=1))


def k_distance_neighbor(p, k):
	dis = distance(p)
	idx = np.argsort(dis)
	for i in range(k + 1, len(idx)):
		if dis[idx[i]] <= dis[idx[i - 1]]:
			i += 1
		else:
			break
	idx = idx[1:i]
	return idx, dis[idx]


def local_reachability_density(index):
	rd = 0.
	for i in range(len(kdn[index])):
		rd += min(dis[index][i], dis[kdn[index][i]][-1])
	return len(kdn[index]) * 1.0 / rd


def local_outlier_factor(k, threshold):
	if k >= len(data):
		print('Wrong k!')
		exit()

	# get k neighbor index and distance
	global kdn, dis
	for i, d in enumerate(data):
		idx, distance = k_distance_neighbor(d, k)
		# index shows the order of k neighbors
		kdn.append(idx)
		# distance shows the distance of k neighbors
		dis.append(distance)

	# get lrd
	global lrd
	for i in range(len(data)):
		lrd[i] = local_reachability_density(i)

	# get lof
	global lof
	for i in range(len(data)):
		lof[i] = np.sum(lrd[kdn[i]]) * 1.0 / len(kdn[i]) / lrd[i]

	normal = lof < threshold
	outliers = lof >= threshold
	return normal, outliers


if __name__ == "__main__":
	k = 15
	threshold = 1.25
	start_time = time.time()
	normal, outliers = local_outlier_factor(k=k, threshold=threshold)
	print('Time used:', time.time() - start_time)
	print('Outliers:', len(data[outliers]))
	plot_outlier(data[normal], outliers=data[outliers], k=k, threshold=threshold)
