import numpy as np
import torch as torch
from joblib import Parallel, delayed
from numpy import ndarray
from sklearn.model_selection import KFold
from torch import tensor
from tqdm import trange, tqdm

from PCA import project_data_onto_pcs
from linearRegression1 import gen_error_and_weights_given_λ, gen_error_given_λ, \
	regularised_linear_regression_model_weights, squared_error
from loadData import load_data
from train_neural_net import train_neural_net


def gen_error_given_hidden_units(data: ndarray, truth: ndarray, folds: KFold, hidden_units: ndarray) -> ndarray:
	N, M = data.shape

	# Add offset attribute
	data = np.concatenate((np.ones((data.shape[0], 1)), data), 1)
	M = M + 1

	# Error for each network for each fold
	error_train: ndarray = np.zeros((len(hidden_units), folds.n_splits))
	error_test: ndarray = np.empty((len(hidden_units), folds.n_splits))

	data, truth = data.astype(np.float32), truth.astype(np.float32)

	for k, (train_index, test_index) in tqdm(enumerate(folds.split(data, truth)), total=folds.n_splits,
											 desc="ANN outer fold", unit="folds"):
		# Extract training and test set for current CV fold
		data_train: tensor = torch.from_numpy(data[train_index])
		truth_train: tensor = torch.from_numpy(truth[train_index])
		data_test: tensor = torch.from_numpy(data[test_index])
		truth_test: tensor = torch.from_numpy(truth[test_index])

		def calculate_error_given_loop_index(i: int) -> None:
			hidden_unit_count: int = hidden_units[i]
			model = lambda: torch.nn.Sequential(
				torch.nn.Linear(M, hidden_unit_count),
				torch.nn.ReLU(),
				torch.nn.Linear(hidden_unit_count, 1)
			)
			loss_fn = torch.nn.MSELoss()
			best_neural_network, best_final_loss, _ = train_neural_net(model, loss_fn, data_train, truth_train,
																	   n_replicates=3)
			error_train[i, k] = best_final_loss
			error_test[i, k] = loss_fn(best_neural_network(data_test).squeeze(), truth_test).data.numpy()

		Parallel(n_jobs=2)(
			delayed(calculate_error_given_loop_index)(i)
			for i in trange(len(hidden_units), desc="Different hidden units", leave=False, unit="networks")
		)

	error_gen: ndarray = np.mean(error_test, axis=1)
	error_μ_train: ndarray = np.mean(error_train, axis=1)
	return error_gen, error_μ_train


def print_table(errors: ndarray, table_meta_info: ndarray):
	table_string: str = "Outer fold\t\t\tANN\t\t\tLinear regression\t\t\tBaseline\n"
	table_string += "i\t\thᵢ\tEᵢ Test\t\tλᵢ\tEᵢ Test\t\t\tEᵢ Test\n"
	for i in range(errors.shape[0]):
		table_string += f"{i}\t\t{table_meta_info[i, 1]:.4f}\t{errors[i, 2]:.4f}\t\t{table_meta_info[i, 0]:.4f}\t{errors[i, 1]:.4f}\t\t\t{errors[i, 0]:.4f}\n"
	print(table_string)


if __name__ == '__main__':
	(data, class_labels, UPDRS) = load_data("train_data.txt")
	projected_data = project_data_onto_pcs(data, 0.9)
	projected_data = np.concatenate((np.ones((projected_data.shape[0], 1)), projected_data), 1)

	K = 5

	hidden_units = [1, 2, 3, 4, 5]
	λs = np.arange(48.87, 48.94, 0.0001)

	error_val = np.empty((K, 3))
	table_meta_info = np.empty((K, 2))

	CV = KFold(n_splits=K, shuffle=True)
	for i, (train_index, test_index) in enumerate(CV.split(projected_data)):
		data_train, data_test = projected_data[train_index], projected_data[test_index]
		UPDRS_train, UPDRS_test = UPDRS[train_index], UPDRS[test_index]
		CV2 = KFold(n_splits=K, shuffle=True)

		# Baseline
		baseline_models: ndarray = np.empty((CV2.n_splits))
		baseline_error_val: ndarray = np.empty((CV2.n_splits))
		for j, (train_index2, test_index2) in enumerate(CV2.split(data_train)):
			data_train2, data_test2 = data_train[train_index2], data_train[test_index2]
			UPDRS_train2, UPDRS_test2 = UPDRS_train[train_index2], UPDRS_train[test_index2]
			baseline_models[j] = np.mean(UPDRS_train2)
			baseline_error_val[j] = squared_error(UPDRS_test2, np.ones((UPDRS_test2.shape[0])) * baseline_models[j])
		error_val[i, 0] = squared_error(UPDRS_test,
										np.ones((UPDRS_test.shape[0])) * baseline_models[baseline_error_val.argmin()])

		# Regularised linear regression
		(rlr_error_val, _) = gen_error_given_λ(data_train, UPDRS_train, CV2, λs)
		optimal_λ = λs[rlr_error_val.argmin()]
		optimal_rlr = regularised_linear_regression_model_weights(data_train.Transpose() @ data_train,
																	data_train.Transpose() @ UPDRS_train,
																	optimal_λ)
		error_val[i, 1], table_meta_info[i, 0] = squared_error(UPDRS_test, data_test @ optimal_rlr), optimal_λ

		# Artificial neural network
		(ann_error_val, _) = gen_error_given_hidden_units(data_train, UPDRS_train, CV2, hidden_units)
		error_val[i, 2], table_meta_info[i, 1] = ann_error_val.min(), hidden_units[ann_error_val.argmin()]
	print_table(error_val, table_meta_info)
