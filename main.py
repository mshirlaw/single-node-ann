#! /usr/bin/python
# A simple Artificial Neural Network with 6 inputs and a single node in the hidden input layer

from numpy import exp
from numpy import array
from numpy import random
from numpy import dot

class ArtificialNeuralNetwork():

	def __init__(self):
		"""Seeds a random number and creates a 6x1 
		matrix of inital weights for the 6 inputs"""
		random.seed(1)
		self.input_weights = 2 * random.random((6,1)) - 1

	def __sigmoid(self, x):
		"""The sigmoid function is a simple activation function
		used to give a value for x in the range 0-1"""
		return 1 / (1 + exp(-x))
	
	def __ddx_sigmoid(self, y):
		"""The first derivative of the sigmoid 
		function given as f(x)' = f(x) * (1 - f(x))"""
		return y * (1 - y)
	
	def train(self, x_values, y_values, iterations):
		"""Train the neural network for n iterations"""
		for iteration in xrange(iterations):
			# apply the sigmoid function to the weighted sum of inputs
			y_calc = self.predict(x_values)
			# calculate the error by comparing output to expected output
			y_error = y_values - y_calc
			# apply the first derivative of the sigmoid function to the output & multiply by error (gradient descent)
			# take the dot product of the transformation matrix of the inital x_values & value calculated above
			adjustments = dot(x_values.T, y_error * self.__ddx_sigmoid(y_calc))
			# adjust input weights
			self.input_weights += adjustments
	
	def predict(self, x_values):
		"""Apply the sigmoid function to the weighted sum of inputs"""
		weighted_sum = dot(x_values, self.input_weights)
		return self.__sigmoid(weighted_sum)
		
		
if __name__ == "__main__":
	
	# create the neural network and train with a training set of data
	neural_network = ArtificialNeuralNetwork()
	x_values = array([[0,0,0,0,0,1],[0,0,0,0,1,1],[0,0,0,1,1,1],[0,0,1,1,1,1]])
	y_values = array([[0,0,1,1]]).T
	neural_network.train(x_values, y_values, 10000)
	
	# use two additional rows not in the training set and predict the result
	# in this case the ANN should predict 1 if there are more 1s in the input row than 0s
	print 1 if neural_network.predict(array([0, 0, 0, 0, 0, 0])) > 0.5 else 0
	print 1 if neural_network.predict(array([1, 1, 1, 1, 1, 1])) > 0.5 else 0
