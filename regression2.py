import matplotlib.pyplot as plt
import numpy as np
import torch as torch
from joblib import Parallel, delayed
from numpy import ndarray
from sklearn.model_selection import KFold
from torch import Tensor, float32, float64, matmul, transpose
from tqdm import trange, tqdm

from PCA import project_data_onto_pcs
from loadData import load_data
from train_neural_net import device, train_neural_net

def ann_creator(input_size: int, hidden_unit_count: int):
	if (hidden_unit_count < 1):
		raise ValueError("hidden_unit_count must be at least 1")
	return lambda: torch.nn.Sequential(
		torch.nn.Linear(input_size, hidden_unit_count),
		torch.nn.ReLU(),
		torch.nn.Linear(hidden_unit_count, 1)
	)


def gen_error_given_hidden_units(data: Tensor, truth: Tensor, folds: KFold, hidden_units: list[int]) -> (Tensor, Tensor):
	# Error for each network for each fold
	error_train: Tensor = torch.zeros(len(hidden_units), folds.n_splits, device=device)
	error_test: Tensor = torch.empty(len(hidden_units), folds.n_splits, device=device)

	models = [ann_creator(data.shape[1], h) for h in hidden_units]

	for k, (train_index, test_index) in enumerate(
			tqdm(folds.split(data, truth), total=folds.n_splits, desc="ANN outer fold", unit="folds")):
		data_train, truth_train = data[train_index], truth[train_index]
		data_test, truth_test = data[test_index], truth[test_index]

		def calculate_error_given_hidden_unit_index(i: int) -> None:
			loss_fn = torch.nn.MSELoss()
			best_neural_network, train_error, _ = train_neural_net(models[i],
																		 loss_fn,
																		 data_train,
																		 truth_train,
																		 n_replicates=5)
			test_error = loss_fn(best_neural_network(data_test).squeeze(), truth_test).data
			error_train[i, k], error_test[i, k] = train_error, test_error

		Parallel(n_jobs=min(len(hidden_units), 5), require='sharedmem')(
			delayed(calculate_error_given_hidden_unit_index)(i)
			for i in range(len(hidden_units))
		)
	return torch.mean(error_test, dim=1), torch.mean(error_train, dim=1)  # Error_gen and Error_train_μ


def print_table(errors: ndarray, table_meta_info: ndarray):
	table_string: str = "Outer fold\t\t\tANN\t\t\tLinear regression\t\t\tBaseline\n"
	table_string += "i\t\thᵢ\tEᵢ Test\t\tλᵢ\tEᵢ Test\t\t\tEᵢ Test\n"
	for i in range(errors.shape[0]):
		table_string += f"{i}\t\t{table_meta_info[i, 1]:.0f}\t{errors[i, 2]:.7f}\t\t{table_meta_info[i, 0]:.7f}\t{errors[i, 1]:.7f}\t\t\t{errors[i, 0]:.7f}\n"
	print(table_string)

def tensor_squared_error(pred: Tensor, truth: Tensor) -> float64:
	return Tensor.square(pred - truth).mean()

def regularised_linear_regression_model_weights(dataTdata: Tensor, dataTtruth: Tensor, λ: float = 0, *, out: Tensor = None) -> Tensor:
	λDiagonalMatrix: Tensor = λ * torch.eye(dataTdata.shape[0], device=device)
	λDiagonalMatrix[0, 0] = 0  # Do not regularize the bias term
	return torch.linalg.solve(dataTdata + λDiagonalMatrix, dataTtruth, out=out)

def gen_error_given_λ(data: Tensor, truth: Tensor, folds: KFold, λs: Tensor) -> (Tensor, Tensor):
	N, M = data.shape

	# Error for each value of λ for each fold
	error_train: Tensor = torch.empty(len(λs), folds.n_splits, device=device)
	error_test: Tensor = torch.ones(len(λs), folds.n_splits, device=device)

	max_batch_size: int = 70000
	#λdiagonal_matricies: Tensor = torch.empty(len(λs), M, M, device=device)

	for k, (train_index, test_index) in enumerate(folds.split(data, truth)):
		# Extract training and test set for current CV fold
		data_train, truth_train = data[train_index], truth[train_index]
		data_test, truth_test = data[test_index], truth[test_index]

		eye_without_first: Tensor = torch.eye(M, device=device)
		eye_without_first[0, 0] = 0  # Do not regularize the bias term
		w_rlrs: Tensor = torch.linalg.solve(
			(data_train.T @ data_train) + λs[:, None, None] * eye_without_first,
			data_train.T @ truth_train).T


		Ntrain, Ntest = len(train_index), len(test_index)
		error_calc_buffer: Tensor = torch.empty(Ntrain, max_batch_size, device=device)
		for i in range(0, w_rlrs.shape[1], max_batch_size):
			batch_end: int = min(i + max_batch_size, w_rlrs.shape[1])
			batch_size: int = batch_end - i

			buffer_train_slice: Tensor = error_calc_buffer[:Ntrain, :batch_size]
			matmul(data_train, w_rlrs[:, i:batch_end], out=buffer_train_slice)
			torch.sub(buffer_train_slice, truth_train[:, None], out=buffer_train_slice)
			torch.square(buffer_train_slice, out=buffer_train_slice)
			error_train[i:batch_end, k] = buffer_train_slice.mean(dim=0)

			buffer_test_slice: Tensor = error_calc_buffer[:Ntest, :batch_size]
			matmul(data_test,  w_rlrs[:, i:batch_end], out=buffer_test_slice)
			torch.sub(buffer_test_slice, truth_test[:, None], out=buffer_test_slice)
			torch.square(buffer_test_slice, out=buffer_test_slice)
			error_test[i:batch_end, k] = buffer_test_slice.mean(dim=0)

	error_gen: Tensor = torch.mean(error_test, dim=1)
	error_μ_train: Tensor = torch.mean(error_train, dim=1)
	return error_gen, error_μ_train


if __name__ == '__main__':
	(data, class_labels, UPDRS) = load_data("train_data.txt")
	projected_data = project_data_onto_pcs(data, 0.9)
	projected_data = np.concatenate((np.ones((projected_data.shape[0], 1)), projected_data), 1)
	projected_data: Tensor = torch.from_numpy(projected_data).type(torch.FloatTensor).to(device)
	UPDRS: Tensor = torch.from_numpy(UPDRS).to(device).type(torch.FloatTensor).to(device)

	K: int = 10

	hidden_units: list[int] = [h for h in range(1, 8)]
	λs: Tensor = torch.arange(48.87, 48.94, 0.0000001, device=device)

	error_val = torch.empty(K, 3, device=device)
	table_meta_info = torch.empty(K, 2, device=device)

	CV = KFold(n_splits=K, shuffle=True)
	for i, (train_index, test_index) in enumerate(tqdm(CV.split(projected_data), desc="Outer fold", total=CV.n_splits, unit="fold")):
		data_train, data_test = projected_data[train_index], projected_data[test_index]
		UPDRS_train, UPDRS_test = UPDRS[train_index], UPDRS[test_index]
		CV2 = KFold(n_splits=K, shuffle=True)

		# Baseline
		baseline_models: Tensor = torch.empty(CV2.n_splits, device=device)
		baseline_error_val: Tensor = torch.empty(CV2.n_splits, device=device)
		for j, (train_index2, test_index2) in enumerate(CV2.split(data_train)):
			data_train2, data_test2 = data_train[train_index2], data_train[test_index2]
			UPDRS_train2, UPDRS_test2 = UPDRS_train[train_index2], UPDRS_train[test_index2]
			baseline_models[j] = torch.mean(UPDRS_train2)
			baseline_error_val[j] = tensor_squared_error(torch.ones(UPDRS_test2.shape[0], device=device) * baseline_models[j], UPDRS_test2)
		error_val[i, 0] = tensor_squared_error(
			torch.ones(UPDRS_test.shape[0], device=device) * baseline_models[baseline_error_val.argmin()],
			UPDRS_test)

		# Regularised linear regression
		(rlr_error_val, _) = gen_error_given_λ(data_train, UPDRS_train, CV2, λs)
		optimal_λ = λs[rlr_error_val.argmin()]
		optimal_rlr = regularised_linear_regression_model_weights(matmul(data_train.T, data_train),
																	matmul(data_train.T, UPDRS_train),
																	optimal_λ)
		error_val[i, 1], table_meta_info[i, 0] = tensor_squared_error(matmul(data_test, optimal_rlr), UPDRS_test), optimal_λ

		# Artificial neural network
		(ann_error_val, _) = gen_error_given_hidden_units(data_train, UPDRS_train, CV2, hidden_units)
		optimal_h = hidden_units[ann_error_val.argmin()]
		loss_function = torch.nn.MSELoss()
		optimal_ann, _, _ = train_neural_net(ann_creator(data_train.shape[1], optimal_h),
											 loss_function,
											 data_train,
											 UPDRS_train,
											 n_replicates = 10)
		error_val[i, 2] = loss_function(optimal_ann(data_test).squeeze(), UPDRS_test).data
		table_meta_info[i, 1] = optimal_h
	print_table(error_val, table_meta_info)




#  This was used in selecting the range of hidden_units (h)
# CV = KFold(n_splits=K, shuffle=True)
# hs: list[int] = [h for h in range(1, 10)]
# error_gen_h, error_train_h_μ = gen_error_given_hidden_units(projected_data, UPDRS, CV, hs)
# plt.figure(4, figsize=(23.4, 16.5))
# plt.plot(hs, Tensor.cpu(error_gen_h.T).numpy(), 'r.-', hs, Tensor.cpu(error_train_h_μ.T).numpy(), 'b.-')
# plt.xlabel('Number of hidden units (h)')
# plt.ylabel('Squared error (cross-validation)')
# plt.legend(['Generalisation error', 'Train error'])
# plt.show()