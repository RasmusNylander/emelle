import numpy as np
import pandas as pd
import sklearn.linear_model
from scipy.linalg import svd
from numpy import double, ndarray
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer


# attrib_group = ['ID', 'jitter', 'jitter', 'jitter', 'jitter', 'jitter', 'shimmer', 'shimmer', 'shimmer', 'shimmer', 'shimmer', 'shimmer', 'harmon', 'harmon', 'harmon', 'pitch', 'pitch', 'pitch', 'pitch', 'pitch', 'pulsing', 'pulsing', 'pulsing', 'pulsing', 'voicing', 'voicing', 'voicing', 'updrs', 'class']
from loadData import load_data

groups: dict[str, int] = {
	'id': 1, 'jitter': 5, 'shimmer': 6, 'harmonicity': 3, 'pitch': 5, 'pulsing': 4, 'voicing': 3, 'updrs': 1, 'label': 1
}


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


def sum_of_squares(truth: ndarray, predictions: ndarray) -> double:
	difference: ndarray = truth-predictions
	return (difference.T*difference).sum()


def squared_error(truth: ndarray, predictions: ndarray) -> double:
	return np.square(truth - predictions).mean(axis=0)


def regularised_linear_regression_model_weights(dataTdata: ndarray, dataTtruth: ndarray, λ: float = 0) -> ndarray:
	λDiagonalMatrix: ndarray = λ * np.eye(dataTdata.shape[0])
	λDiagonalMatrix[0, 0] = 0  # Do not regularize the bias term
	w_rlr = np.linalg.solve(dataTdata + λDiagonalMatrix, dataTtruth).squeeze()
	return w_rlr


def gen_error_given_λ(data: ndarray, truth: ndarray, folds: KFold, λs: ndarray) -> (ndarray, ndarray):
	N, M = data.shape

	# Add offset attribute
	data = np.concatenate((np.ones((data.shape[0], 1)), data), 1)
	M = M + 1

	# Error for each value of λ for each fold
	error_train: ndarray = np.empty((len(λs), folds.n_splits))
	error_test: ndarray = np.empty((len(λs), folds.n_splits))

	w_rlr: ndarray = np.empty(M)  # weights for regularized logistic regression

	for k, (train_index, test_index) in enumerate(folds.split(data, truth)):
		# Extract training and test set for current CV fold
		data_train: ndarray = data[train_index]
		truth_train: ndarray = truth[train_index]
		data_test: ndarray = data[test_index]
		truth_test: ndarray = truth[test_index]

		dataTdata: ndarray = data_train.T @ data_train
		dataTtruth: ndarray = data_train.T @ truth_train

		for i in range(0, len(λs)):
			w_rlr = regularised_linear_regression_model_weights(dataTdata, dataTtruth, λs[i])
			error_train[i, k] = squared_error(truth_train, data_train @ w_rlr)
			error_test[i, k] = squared_error(truth_test, data_test @ w_rlr)

	error_gen: ndarray = np.mean(error_test, axis=1)
	error_μ_train: ndarray = np.mean(error_train, axis=1)
	return (error_gen, error_μ_train)

if __name__ == '__main__':
	# Load data
	(data, class_labels, UPDRS) = load_data("train_data.txt")

	# PCA
	(U, Σ, Vh) = svd(data)
	V = Vh.T

	ρ: ndarray = explained_var(Σ)
	threshold = 0.9
	plot_explained_variance(ρ, threshold, "PCA_explained_variance.pdf")
	num_pc_to_threshold = (np.cumsum(ρ) < threshold).sum() + 1
	print(f"Acceptable threshold: {threshold}\nRequired number of components: {num_pc_to_threshold}")

	data_projected = data @ V[:, :num_pc_to_threshold]  # Data projected onto {num_pc_to_threshold} pr
	plot_data_projected_unto_principal_components(data @ V[:, :4], class_labels)


	# Regularised Linear Regression, using 10-fold cross validation
	folds: KFold = KFold(10, shuffle=True)
	λs: ndarray = np.power(10, np.arange(-1, 7, 0.001))
	#optimal_λ: ndarray = np.empty(folds.n_splits)

	(error_gen_λ, error_train_μ_λ) = gen_error_given_λ(data_projected, UPDRS, folds, λs)

	min_λ_index = np.argmin(error_gen_λ, axis=0)
	rlr_model_optimal_λ = regularised_linear_regression_model_weights(data_projected.T @ data_projected, data_projected.T @ UPDRS, λs[min_λ_index])
	print(f"Optimal λ: {λs[min_λ_index]}\nModel weights: {rlr_model_optimal_λ}")

	# Plot Generalisation error and mean train error as a function of λ
	plt.figure(3, figsize=(23.4, 16.5))
	plt.loglog(λs, error_gen_λ.T, 'r.-', λs, error_train_μ_λ.T, 'b.-')
	plt.axvline(x=λs[min_λ_index], color='k', linestyle='--')
	#plt.xticks(plt.xticks()[0] + [λs[min_λ_index]])
	plt.xlabel('Regularization factor (λ)')
	plt.ylabel('Squared error (cross - validation)')
	plt.legend(['Generalisation error', 'Train error'])
	plt.grid()



	plt.show()




# Old, probably ignore
def estimated_generalisation_error(model_generator_λ, cost_function, data: ndarray, truth: ndarray, folds: KFold):
	test_error = np.empty((folds.n_splits, 1))
	train_error = np.empty((folds.n_splits, 1))
	generalisation_error = 0
	split_num = 0
	for train_index, test_index in folds.split(data, truth):
		train_data: ndarray = data[train_index]
		train_truth: ndarray = truth[train_index]
		test_data: ndarray = data[test_index]
		test_truth: ndarray = truth[test_index]

		model = model_generator_λ()
		model.fit(train_data, train_truth)
		train_predict = model.predict(train_data).T
		test_predict = model.predict(test_data).T

		train_error[split_num] = cost_function(train_truth, train_predict)/len(train_truth)
		test_error[split_num] = cost_function(test_truth, test_predict)/len(test_truth)
		generalisation_error += test_error[split_num]

		split_num += 1
	generalisation_error /= folds.n_splits
	return generalisation_error