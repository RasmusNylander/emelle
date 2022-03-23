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

names = ['id', 'jitter_local', 'jitter_local_absolute', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_local',
		 'shimmer_local_dp', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'AC', 'NTH', 'HTN',
		 'median_pitch', 'mean_pitch', 'standard_deviation', 'minimum_pitch', 'maximum_pitch', 'number_of_pulses',
		 'number_of_periods', 'mean_period', 'standard_deviation_of_period', 'fraction_of_locally_unvoiced_frames',
		 'number_of_voice_breaks', 'degree_of_voice_breaks', 'UPDRS', 'class_information']
# attrib_group = ['ID', 'jitter', 'jitter', 'jitter', 'jitter', 'jitter', 'shimmer', 'shimmer', 'shimmer', 'shimmer', 'shimmer', 'shimmer', 'harmon', 'harmon', 'harmon', 'pitch', 'pitch', 'pitch', 'pitch', 'pitch', 'pulsing', 'pulsing', 'pulsing', 'pulsing', 'voicing', 'voicing', 'voicing', 'updrs', 'class']
groups: dict[str, int] = {
	'id': 1, 'jitter': 5, 'shimmer': 6, 'harmonicity': 3, 'pitch': 5, 'pulsing': 4, 'voicing': 3, 'updrs': 1, 'label': 1
}


def get_data(csv_path: str) -> DataFrame:
	dataframe: DataFrame = pd.read_csv(csv_path, names=names)
	# dataframe: DataFrame = dataframe.rolling(26).mean().iloc[::26, :]  # Only relevant for test dataset
	# dataframe: DataFrame = dataframe.iloc[:, 1:-2]  # drop the subject ID, UPDRS, and class label

	return dataframe


def normalised_data(data: ndarray) -> ndarray:
	data: ndarray = data - np.ones((len(data), 1)) * np.mean(data, axis=0)
	data: ndarray = data * (1 / np.std(data, 0))
	return data


def explained_var(Σ: ndarray) -> ndarray:
	ΣΣ: ndarray = Σ * Σ
	return ΣΣ / ΣΣ.sum()


def one_out_of_k_encode_sounds(data: ndarray, categorySizes: list[int]) -> ndarray:
	numberOfRepetitions = int(data.shape[0] / np.sum(categorySizes))
	if numberOfRepetitions < 0:
		return dataframe

	sound_number = []
	for id, num in enumerate(categorySizes):
		sound_number.extend(id for _ in range(num))

	categorised = []
	for i in range(numberOfRepetitions):
		categorised.extend(sound_number)

	one_out_of_k_encoding: DataFrame = pd.get_dummies(categorised)
	return np.append(data, one_out_of_k_encoding.to_numpy(), axis=1)


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


def estimated_generalisation_error(model_generator_λ, data: ndarray, truth: ndarray, folds: KFold):
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
		train_error[split_num] = sum_of_squares(train_truth, train_predict)/len(train_truth)
		test_error[split_num] = sum_of_squares(test_truth, test_predict)/len(test_truth)
		generalisation_error += test_error[split_num]
		split_num += 1
	generalisation_error /= folds.n_splits
	return generalisation_error

def sum_of_squares(truth: ndarray, predictions: ndarray) -> double:
	difference: ndarray = truth-predictions
	return (difference.T*difference).sum()

if __name__ == '__main__':
	plt.close('all')
	savePlots: bool = False
	dataframe: DataFrame = get_data("train_data.txt")
	class_labels: ndarray = dataframe.iloc[:, -1].to_numpy()
	UPDRS: ndarray = dataframe.iloc[:, -2].to_numpy()
	dataframe: DataFrame = dataframe.iloc[:, 1:-2]  # drop the subject ID, UPDRS, and class label
	data: ndarray = normalised_data(dataframe.to_numpy())
	data: ndarray = one_out_of_k_encode_sounds(data, [3, 10, 4, 9])
	(U, Σ, Vh) = svd(data)
	V = Vh.T

	ρ: ndarray = explained_var(Σ)
	threshold = 0.9
	plot_explained_variance(ρ, threshold, "PCA_explained_variance.pdf")
	num_pc_to_threshold = (np.cumsum(ρ) < threshold).sum() + 1
	print(f"Acceptable threshold: {threshold}\nRequired number of components: {num_pc_to_threshold}")

	data_projected = data @ V[:, :num_pc_to_threshold]  # Data projected onto {num_pc_to_threshold} pr
	plot_data_projected_unto_principal_components(data @ V[:, :4], class_labels)

	λ = [0, 0.0001, 0.001, 0.01, 0.1, 1, 5, 29752]
	generalisation_error = []
	folds: KFold = KFold(10, shuffle=True)
	for i, λambda in enumerate(λ):
		generalisation_error.append(
			estimated_generalisation_error(
				lambda: LinearRegression(),
				data_projected, UPDRS, folds)
		)


	plt.show()




