import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_neural_net(model, loss_fn, data, truth,
					 n_replicates=3, max_iter=10000, tolerance=1e-6):
	"""
	Train a neural network with PyTorch based on a training set consisting of
	observations X and class y. The model and loss_fn inputs define the
	architecture to train and the cost-function update the weights based on,
	respectively.

	Usage:
		Assuming loaded dataset (X,y) has been split into a training and
		test set called (X_train, y_train) and (X_test, y_test), and
		that the dataset has been cast into PyTorch tensors using e.g.:
			X_train = torch.tensor(X_train, dtype=torch.float)
		Here illustrating a binary classification example based on e.g.
		M=2 features with H=2 hidden units:

		>>> # Define the overall architechture to use
		>>> model = lambda: torch.nn.Sequential(
					torch.nn.Linear(M, H),  # M features to H hiden units
					torch.nn.Tanh(),        # 1st transfer function
					torch.nn.Linear(H, 1),  # H hidden units to 1 output neuron
					torch.nn.Sigmoid()      # final tranfer function
					)
		>>> loss_fn = torch.nn.BCELoss() # define loss to use
		>>> net, final_loss, learning_curve = train_neural_net(model,
													   loss_fn,
													   X=X_train,
													   y=y_train,
													   n_replicates=3)
		>>> y_test_est = net(X_test) # predictions of network on test set
		>>> # To optain "hard" class predictions, threshold the y_test_est
		>>> See exercise ex8_2_2.py for indepth example.

		For multi-class with C classes, we need to change this model to e.g.:
		>>> model = lambda: torch.nn.Sequential(
							torch.nn.Linear(M, H), #M features to H hiden units
							torch.nn.ReLU(), # 1st transfer function
							torch.nn.Linear(H, C), # H hidden units to C classes
							torch.nn.Softmax(dim=1) # final tranfer function
							)
		>>> loss_fn = torch.nn.CrossEntropyLoss()

		And the final class prediction is based on the argmax of the output
		nodes:
		>>> y_class = torch.max(y_test_est, dim=1)[1]

	Args:
		model:          A function handle to make a torch.nn.Sequential.
		loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary
						binary classification, torch.nn.CrossEntropyLoss() for
						multiclass classification, or torch.nn.MSELoss() for
						regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
		n_replicates:   An integer specifying number of replicates to train,
						the neural network with the lowest loss is returned.
		max_iter:       An integer specifying the maximum number of iterations
						to do (default 10000).
		tolerenace:     A float describing the tolerance/convergence criterion
						for minimum relative change in loss (default 1e-6)


	Returns:
		A list of three elements:
			best_net:       A trained torch.nn.Sequential that had the lowest
							loss of the trained replicates
			final_loss:     An float specifying the loss of best performing net
			learning_curve: A list containing the learning curve of the best net.

	"""
	networks: ndarray = np.empty(n_replicates, dtype=object)
	final_losses: Tensor = torch.empty(n_replicates, device=device)
	learning_curves: Tensor = torch.empty(n_replicates, max_iter, device=device).type(torch.FloatTensor)
	for i in range(n_replicates):
		networks[i], final_losses[i], learning_curves[i, :] = train_net(model().to(device), max_iter, tolerance, data, truth, loss_fn)

	#from joblib import Parallel, delayed
	#networks, loss, learning_curves = zip(*Parallel(n_jobs=min(5, n_replicates))(
	#	delayed(train_net)(model, max_iter, tolerance, X, y, loss_fn)
	#	for i in range(n_replicates)
	#))

	#networks, loss, learning_curves = zip(*Parallel(n_jobs=1)(
	#	delayed(train_net)(model, max_iter, tolerance, X, y, loss_fn)
	#	for i in range(n_replicates)
	#))


	# Return the best curve along with its final loss and learing curve
	return networks[torch.argmin(final_losses)], torch.min(final_losses), learning_curves[torch.argmin(final_losses)]


def train_net(network, max_iterations, tolerance, data, truth, loss_function):
	import torch
	learning_curve: Tensor = torch.zeros(max_iterations, device=device)  # setup storage for loss at each step
	old_loss = 1e6
	torch.nn.init.xavier_uniform_(network[0].weight).to(device)
	torch.nn.init.xavier_uniform_(network[2].weight).to(device)
	optimizer = torch.optim.Adam(network.parameters())
	for i in range(max_iterations):
		y_est = network(data).squeeze() # forward pass, predict labels on training set
		loss = loss_function(y_est, truth)  # determine loss
		loss_value = loss.data
		learning_curve[i] = loss_value # record loss for later display

		# Convergence check, see if the percentual loss decrease is within
		# tolerance:
		p_delta_loss = torch.abs(loss_value - old_loss) / old_loss
		if p_delta_loss < tolerance: break
		old_loss = loss_value

		# do backpropagation of loss and optimize weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return network, loss_value, learning_curve