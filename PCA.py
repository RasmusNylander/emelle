import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.linalg import svd

def explained_var(Σ: ndarray) -> ndarray:
	ΣΣ: ndarray = Σ * Σ
	return ΣΣ / ΣΣ.sum()

def plot_explained_variance(ρ: ndarray, threshold = 0.9, save_to_file: str = ""):
	plot_range = range(1, len(ρ) + 1)
	f = plt.figure()
	plt.plot(plot_range, ρ, 'x-')
	plt.plot(plot_range, np.cumsum(ρ), 'o-')
	plt.plot([1, len(ρ)], [threshold, threshold], 'k--')
	plt.xlabel('Principal component')
	plt.ylabel('Variance explained')
	plt.legend(['Individual', 'Cumulative', 'Threshold'])
	plt.grid()
	if save_to_file != '':
		plt.savefig(save_to_file)
	return f


def plot_data_projected_unto_principal_components(projected_data: ndarray, class_labels: ndarray):
	classes: ndarray = np.unique(class_labels)
	num_principal_components = projected_data.shape[1]
	num_rows = 2
	num_columns = int(np.ceil((num_principal_components-1)*2/num_rows))
	fig, subplots = plt.subplots(num_rows, num_columns)
	plt.title('Data projected onto Principal components')
	for d1 in range(num_principal_components):
		for d2 in range(d1 + 1, num_principal_components):
			index = d1 + d2 - 1
			for klass in classes:
				class_mask: ndarray = (class_labels == klass)
				subplots[int(np.floor(index / num_columns)), index % num_columns].plot(data_projected[class_mask, d1], data_projected[class_mask, d2], 'o')
			#subplots[d1, d2].plt.legend(['With Parkinson\'s', 'Without Parkinson\'s'])
			#subplots[d1, d2].xlabel(f'PC{d1}')
			subplots[(d1 + d2 - 1) % num_rows, (d1 + d2 - 1) % num_columns ].set(xlabel=f'PC{d1}', ylabel=f'PC{d2}')


def project_data_onto_pcs(data: ndarray, threshold) -> ndarray:
	if threshold < 0 or threshold > 1:
		raise ValueError('Threshold must be between 0 and 1')

	(U, Σ, Vh) = svd(data)
	V = Vh.T

	ρ: ndarray = explained_var(Σ)
	num_pc_to_threshold = (np.cumsum(ρ) < threshold).sum() + 1
	data_projected = data @ V[:, :num_pc_to_threshold]  # Data projected onto {num_pc_to_threshold} components
	#plot_explained_variance(ρ, threshold, "PCA_explained_variance.pdf")
	#print(f"Acceptable threshold: {threshold}\nRequired number of components: {num_pc_to_threshold}")
	#plot_data_projected_unto_principal_components(data @ V[:, :4], class_labels)
	return data_projected