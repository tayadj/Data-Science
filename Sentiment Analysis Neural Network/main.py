import numpy as np
import re

class Model:

    def __init__(self, hidden_dimension = 64, data_path = './data/sentences.npz'):

        self.activation = lambda x: np.exp(x) / sum(np.exp(x))
        self.activation_derivative = lambda x: (1 - x ** 2)
        self.loss = lambda x: -np.log(x)
        
        self.load_data(data_path)
        self.load_vocabulary()
        self.load_network(hidden_dimension)

    def normalize_text(self, text):

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        return text

    def convert_text(self, text):
        
        text = self.normalize_text(text)
        inputs = []

        for word in text.split(' '):

            value = np.zeros((self.vocabulary_size, 1))
            value[self.word_to_index[word]] = 1
            inputs.append(value)

        return inputs

    def load_data(self, path):

        with np.load(path) as loaded:
            
            raw_data = dict(zip(loaded['sentences'], loaded['labels']))
            self.data = { self.normalize_text(text) : label for text, label in raw_data.items() }

    def load_vocabulary(self):

        self.vocabulary = list(set([word for text in self.data.keys() for word in text.split(' ')]))
        self.vocabulary_size = len(self.vocabulary)

        self.word_to_index = { word : index for index, word in enumerate(self.vocabulary) }
        self.index_to_word = { index : word for index, word in enumerate(self.vocabulary) }

    def load_network(self, dimension_hidden):
        
        self.dimension_input = self.vocabulary_size
        self.dimension_output = 2
        self.dimension_hidden = dimension_hidden

        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (self.dimension_hidden, self.dimension_input))
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (self.dimension_output, self.dimension_hidden))
        self.weights_hidden_to_hidden = np.random.uniform(-0.5, 0.5, (self.dimension_hidden, self.dimension_hidden))
        self.bias_hidden = np.zeros((self.dimension_hidden, 1))
        self.bias_output = np.zeros((self.dimension_output, 1))

    def forward(self, inputs):

        hidden = np.zeros((self.weights_hidden_to_hidden.shape[0], 1))

        self.recent_inputs = inputs
        self.recent_hidden = { 0: hidden }

        for index, vector in enumerate(inputs):

            hidden = np.tanh(self.weights_input_to_hidden @ vector + self.weights_hidden_to_hidden @ hidden + self.bias_hidden)
            
            self.recent_hidden[index + 1] = hidden

        output = self.weights_hidden_to_output @ hidden + self.bias_output

        return hidden, output

    def backward(self, gradient_output, learning_rate):

        size = len(self.recent_inputs)

        gradient_weights_hidden_to_output = gradient_output @ self.recent_hidden[size].T
        gradient_bias_output = gradient_output

        gradient_weights_hidden_to_hidden = np.zeros(self.weights_hidden_to_hidden.shape)
        gradient_weights_input_to_hidden = np.zeros(self.weights_input_to_hidden.shape)
        gradient_bias_hidden = np.zeros(self.bias_hidden.shape)

        gradient_hidden = self.weights_hidden_to_output.T @ gradient_output

        for time in reversed(range(size)):

            delta = self.activation_derivative(self.recent_hidden[time + 1]) * gradient_hidden
            gradient_bias_hidden += delta
            gradient_weights_hidden_to_hidden += delta @ self.recent_hidden[time].T
            gradient_weights_input_to_hidden += delta @ self.recent_inputs[time].T
            gradient_hidden = self.weights_hidden_to_hidden @ delta

        for gradient in [gradient_weights_input_to_hidden, gradient_weights_hidden_to_hidden, gradient_weights_hidden_to_output, gradient_bias_hidden, gradient_bias_output]:
            
            np.clip(gradient, -1, 1, out = gradient)

        self.weights_hidden_to_output -= learning_rate * gradient_weights_hidden_to_output
        self.weights_hidden_to_hidden -= learning_rate * gradient_weights_hidden_to_hidden
        self.weights_input_to_hidden -= learning_rate * gradient_weights_input_to_hidden
        self.bias_output -= learning_rate * gradient_bias_output
        self.bias_hidden -= learning_rate * gradient_bias_hidden 
        
    def process(self, data, learning_rate, learn = True):

        loss = 0
        correct = 0

        for sentence, label in self.data.items():
           
            inputs = self.convert_text(sentence)
            target = int(label)

            hidden, output = self.forward(inputs)

            probabilities = self.activation(output)
            
            loss += self.loss(probabilities[target])[0]
            correct += int(np.argmax(probabilities) == target)

            if learn:
            
                probabilities[target] -= 1
                self.backward(probabilities, learning_rate)

        return loss / len(data), correct / len(data)

    def learn(self, epochs = 100, learning_rate = 0.001):

        for epoch in range (epochs):

            coefficient_loss, coefficient_accuracy = self.process(self.data, learning_rate)

            if epoch % 10 == 9:

                print(f'Epoch {epoch+1}:')
                print(f'Loss: {round(coefficient_loss * 100, 3)}%')
                print(f'Accuracy: {round(coefficient_accuracy * 100, 3)}%')
            
    def predict(self, sentence):
        
        inputs = self.convert_text(sentence)

        hidden, output = self.forward(inputs)

        probabilities = self.activation(output)
        prediction = np.argmax(probabilities) 

        if prediction == 0:

            print('This is negative sentence.')

        else:

            print('This is positive sentence.')

    def save(self, path = './data/model.npz'):

        np.savez(path, weights_input_to_hidden = self.weights_input_to_hidden, weights_hidden_to_output = self.weights_hidden_to_output, weights_hidden_to_hidden = self.weights_hidden_to_hidden, bias_hidden = self.bias_hidden, bias_output = self.bias_output)

    def load(self, path = './data/model.npz'):

        with np.load('./data/model.npz') as loaded:

            self.weights_input_to_hidden = loaded['weights_input_to_hidden']
            self.weights_hidden_to_output = loaded['weights_hidden_to_output']
            self.weights_hidden_to_hidden = loaded['weights_hidden_to_hidden']
            self.bias_hidden = loaded['bias_hidden']
            self.bias_output = loaded['bias_output']
