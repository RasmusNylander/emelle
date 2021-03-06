import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tolerance: Tensor = torch.Tensor([1e-6]).to(device=device)

def train_neural_net(model, loss_fn, data: Tensor, truth: Tensor,
					 n_replicates=3, max_iter=10000):
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

	best_net, best_loss, curve_of_best = None, np.inf, []
	for i in range(n_replicates):
		network, final_loss, learning_curve = train_net(model().to(device=device), max_iter, tolerance, data, truth, loss_fn)
		if final_loss < best_loss:
			best_net, best_loss, curve_of_best = network, final_loss, learning_curve

	# Return the best curve along with its final loss and learing curve
	return best_net, best_loss, curve_of_best


def train_net(network, max_iterations: int, tolerance: Tensor, data: Tensor, truth: Tensor, loss_function):
	import torch
	learning_curve: Tensor = torch.zeros(max_iterations, device=device)  # setup storage for loss at each step
	old_loss: Tensor = torch.Tensor([np.inf]).to(device=device)  # initialize with large value
	torch.nn.init.xavier_uniform_(network[0].weight).to(device)
	torch.nn.init.xavier_uniform_(network[2].weight).to(device)
	optimizer = torch.optim.Adam(network.parameters())

	for i in range(max_iterations):
		loss = loss_function(network(data).squeeze(), truth)  # determine loss
		loss_value: Tensor = loss.data
		learning_curve[i] = loss_value # record loss for later display

		# Convergence check, see if the percentual loss decrease is within
		# tolerance:
		p_delta_loss = torch.abs(loss_value - old_loss) / old_loss
		if p_delta_loss < tolerance:
			break
		old_loss = loss_value

		# do backpropagation of loss and optimize weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return network, loss_value, learning_curve