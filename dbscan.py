from dataset import get_dataset
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import time

data = get_dataset()
to_index = np.arange(len(data))


def plot_outlier(mark, num_cluster, eps, min_points, min_cluster_size=2):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# outliers
	points = data[mark == -1]
	count = len(points)
	ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='outliers')
	# normal
	cluster = 1
	for i in range(num_cluster + 1):
		points = data[mark == i]
		if len(points) < min_cluster_size:
			ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r')
			count += len(points)
		else:
			ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='cluster %d' % cluster)
			cluster += 1
	print('Outlier:', count)
	ax.set_xlabel('L')
	ax.set_ylabel('F')
	ax.set_zlabel('M')
	ax.legend()
	plt.title('eps=%d, min_points=%d' % (eps, min_points))
	plt.show()


def distance(a, b):
	return np.sqrt(np.sum((a - b) ** 2, axis=1))


def eps_neighbor(p, eps):
	dis = distance(data[p], data)
	# neighbor does not include p itself
	dis[dis <= 0] = eps + 1
	return to_index[dis <= eps]


def dbscan(eps=0.1, min_points=10):
	unvisited = set(range(len(data)))
	mark = np.zeros(len(data))
	num_cluster = 0
	while len(unvisited):
		p = unvisited.pop()
		neighbor = eps_neighbor(p, eps)
		if len(neighbor) >= min_points:
			num_cluster += 1
			mark[p] = num_cluster
			for q in neighbor:
				if q in unvisited:
					unvisited.remove(q)
					q_neighbor = eps_neighbor(q, eps)
					if len(q_neighbor) >= min_points:
						neighbor = np.concatenate((neighbor, q_neighbor))
					if mark[q] != 0:
						mark[q] = num_cluster
		else:
			mark[p] = -1

	return mark, num_cluster


if __name__ == "__main__":
	eps = 25
	min_points = 10
	min_cluster_size = 2
	start_time = time.time()
	mark, num_cluster = dbscan(eps=eps, min_points=min_points)
	print('Time used:', time.time() - start_time)
	plot_outlier(mark, num_cluster, eps, min_points, min_cluster_size=min_cluster_size)
