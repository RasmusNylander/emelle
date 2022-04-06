import numpy as np
from matplotlib import pyplot as plt
from numpy import double, ndarray
from sklearn.model_selection import KFold

from PCA import project_data_onto_pcs
from loadData import load_data


def regularised_linear_regression_model_weights(dataTdata: ndarray, dataTtruth: ndarray, λ: float = 0) -> ndarray:
	λDiagonalMatrix: ndarray = λ * np.eye(dataTdata.shape[0])
	λDiagonalMatrix[0, 0] = 0  # Do not regularize the bias term
	w_rlr = np.linalg.solve(dataTdata + λDiagonalMatrix, dataTtruth).squeeze()
	return w_rlr

def squared_error(truth: ndarray, predictions: ndarray) -> double:
	return np.square(truth - predictions).mean(axis=0)

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
	return error_gen, error_μ_train


if __name__ == '__main__':
	(data, class_labels, UPDRS) = load_data("train_data.txt")
	projected_data = project_data_onto_pcs(data, 0.9)

	# Regularised Linear Regression, using 10-fold cross validation
	K: int = 10
	λs: ndarray = np.power(10, np.arange(-1, 7, 0.001))
	# We repeat it multiple times, and take the average of the results
	# This ensures that if we run it multiple times, we always get more or less the same result.
	repetitions: int = 50
	gen_errors_λ: ndarray = np.empty((len(λs), repetitions))
	train_errors_μ_λ: ndarray = np.empty((len(λs), repetitions))

	for i in range(0, repetitions):
		folds: KFold = KFold(K, shuffle=True)
		# (gen_errors, train_errors_μ) = gen_error_given_λ(projected_data, class_labels, folds, λs)
		(gen_errors_λ[:, i], train_errors_μ_λ[:, i]) = gen_error_given_λ(projected_data, UPDRS, folds, λs)
	generalisation_errors = np.mean(gen_errors_λ, axis=1)
	training_errors: ndarray = np.mean(train_errors_μ_λ, axis=1)

	min_λ_index = np.argmin(generalisation_errors, axis=0)
	rlr_model_optimal_λ = regularised_linear_regression_model_weights(projected_data.T @ projected_data, projected_data.T @ UPDRS, λs[min_λ_index])
	print(f"Optimal λ: {λs[min_λ_index]}\nModel weights: {rlr_model_optimal_λ}")

	# Plot Generalisation error and mean train error as a function of λ
	plt.figure(3, figsize=(23.4, 16.5))
	plt.loglog(λs, generalisation_errors.T, 'r.-', λs, training_errors.T, 'b.-')
	plt.axvline(x=λs[min_λ_index], color='k', linestyle='--')
	#plt.xticks(plt.xticks()[0] + [λs[min_λ_index]])
	plt.xlabel('Regularization factor (λ)')
	plt.ylabel('Squared error (cross - validation)')
	plt.legend(['Generalisation error', 'Train error'])
	plt.grid()

	plt.show()