import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

def read_data(csv_path: str) -> DataFrame:
	names = ['id', 'jitter_local', 'jitter_local_absolute', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_local',
			 'shimmer_local_dp', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'AC', 'NTH', 'HTN',
			 'median_pitch', 'mean_pitch', 'standard_deviation', 'minimum_pitch', 'maximum_pitch', 'number_of_pulses',
			 'number_of_periods', 'mean_period', 'standard_deviation_of_period', 'fraction_of_locally_unvoiced_frames',
			 'number_of_voice_breaks', 'degree_of_voice_breaks', 'UPDRS', 'class_information']
	dataframe: DataFrame = pd.read_csv(csv_path, names=names)
	# dataframe: DataFrame = dataframe.rolling(26).mean().iloc[::26, :]  # Only relevant for test dataset
	# dataframe: DataFrame = dataframe.iloc[:, 1:-2]  # drop the subject ID, UPDRS, and class label
	return dataframe

def normalised_data(data: ndarray) -> ndarray:
	data: ndarray = data - np.ones((len(data), 1)) * np.mean(data, axis=0)
	data: ndarray = data * (1 / np.std(data, 0))
	return data

def one_out_of_k_encode_sounds(data: ndarray, categorySizes: list[int]) -> ndarray:
	numberOfRepetitions = int(data.shape[0] / np.sum(categorySizes))
	if numberOfRepetitions < 0:
		return data

	sound_number = []
	for id, num in enumerate(categorySizes):
		sound_number.extend(id for _ in range(num))

	categorised = []
	for i in range(numberOfRepetitions):
		categorised.extend(sound_number)

	one_out_of_k_encoding: DataFrame = pd.get_dummies(categorised)
	return np.append(data, one_out_of_k_encoding.to_numpy(), axis=1)

def load_data(path: str) -> (ndarray, ndarray, ndarray):
	dataframe: DataFrame = read_data(path)
	class_labels: ndarray = dataframe.iloc[:, -1].to_numpy()
	UPDRS: ndarray = dataframe.iloc[:, -2].to_numpy()
	dataframe: DataFrame = dataframe.iloc[:, 1:-2]  # drop the subject ID, UPDRS, and class label
	data: ndarray = normalised_data(dataframe.to_numpy())
	data: ndarray = one_out_of_k_encode_sounds(data, [3, 10, 4, 9])
	return data, class_labels, UPDRS