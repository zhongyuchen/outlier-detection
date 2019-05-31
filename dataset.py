import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot as plt

data_path = 'consumption_data.xls'


def get_dataset():
	data = np.array(pd.read_excel(io=data_path).values)[:, 1:4]
	print('Size:', len(data))
	print('Sample:', data[0])
	return data


def plot_outlier(data):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:, 0], data[:, 1], data[:, 2])
	ax.set_xlabel('L')
	ax.set_ylabel('F')
	ax.set_zlabel('M')
	ax.legend()
	plt.show()


if __name__ == "__main__":
	data = get_dataset()
	plot_outlier(data)
