import numpy as np
import matplotlib.pyplot as plt

class Model:

	def __init__(self):

		self.dimension_input = 784
		self.dimension_hidden = 28
		self.dimension_output = 10

		self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (self.dimension_hidden, self.dimension_input))
		self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (self.dimension_output, self.dimension_hidden))
		self.bias_input_to_hidden = np.zeros((self.dimension_hidden,1))
		self.bias_hidden_to_output = np.zeros((self.dimension_output,1))

	def data(self):

		with np.load('./data/fashion-mnist.npz') as loaded:

			x_train = loaded['x_train'].astype('float32') / 255.0
			x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
			y_train = loaded['y_train']
			y_train = np.eye(10)[y_train]
			return x_train, y_train
	
	def learn_activation(self, value):
		
		return 1 / (1 + np.exp(-value))

	def learn_loss(self, observed, expected):

		return np.sum((observed - expected) ** 2, axis = 0) / len(observed)

	def learn_forward(self, image):

		hidden_raw = self.weights_input_to_hidden @ image + self.bias_input_to_hidden
		hidden = self.learn_activation(hidden_raw)
	
		output_raw = self.weights_hidden_to_output @ hidden + self.bias_hidden_to_output
		output = self.learn_activation(output_raw)

		return hidden, output

	def learn_backward(self, output, hidden, image, label, learning_rate):
		
		delta_output = output - label
		self.weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
		self.bias_hidden_to_output += -learning_rate * delta_output

		delta_hidden = np.transpose(self.weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
		self.weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
		self.bias_input_to_hidden += -learning_rate * delta_hidden
		
	def learn(self, epochs = 10, learning_rate = 0.001):
		
		images, labels = self.data()

		coefficient_loss = 0.0
		coefficient_accuracy = 0.0
		
		for epoch in range(epochs):
			
			print(f'Epoch #{epoch+1}:')

			for image, label in zip(images, labels):
				
				image = np.reshape(image, (-1, 1))
				label = np.reshape(label, (-1, 1))

				hidden, output = self.learn_forward(image)

				coefficient_loss += self.learn_loss(output, label)[0]
				coefficient_accuracy += int(np.argmax(output) == np.argmax(label))

				self.learn_backward(output, hidden, image, label, learning_rate)

			print(f'Loss: {round((coefficient_loss / images.shape[0]) * 100, 3)}%')
			print(f'Accuracy: {round((coefficient_accuracy / images.shape[0]) * 100, 3)}%')
			coefficient_loss = 0.0
			coefficient_accuracy = 0.0

	def predict(self):

		labels_text = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
		images, labels = self.data()
		index = np.random.randint(0, images.shape[0])
		image = images[index]
		label = labels[index]

		image_reshaped = np.reshape(image, (-1, 1))
		hidden, output = self.learn_forward(image_reshaped)

		predicted_label = np.argmax(output)
		true_label = np.argmax(label)

		plt.imshow(image.reshape(28, 28), cmap='Greys')
		plt.title(f'Predicted: {labels_text[predicted_label]}, True: {labels_text[true_label]}')
		plt.show()

	def save(self, path = './data/model.npz'):

		np.savez(path, weights_input_to_hidden = self.weights_input_to_hidden, 
					   weights_hidden_to_output = self.weights_hidden_to_output,
					   bias_input_to_hidden = self.bias_input_to_hidden,
					   bias_hidden_to_output = self.bias_hidden_to_output)

	def load(self, path = './data/model.npz'):

		with np.load('./data/model.npz') as loaded:

			self.weights_input_to_hidden = loaded['weights_input_to_hidden']
			self.weights_hidden_to_output = loaded['weights_hidden_to_output']
			self.bias_input_to_hidden = loaded['bias_input_to_hidden']
			self.bias_hidden_to_output = loaded['bias_hidden_to_output']

