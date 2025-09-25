import numpy as np

class Perceptron:
    def __init__(self):
        pass

    # Função de ativação: degrau
    def ativacao_funcao(self, x):
        return 1 if x >= 0 else 0

    # Treinamento do perceptron
    def train(self, inputs, outputs, learning_rate=0.1, epochs=100):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Inicialização dos pesos e bias (para x1, x2, x3)
        w1 = np.random.rand(1,)
        w2 = np.random.rand(1,)
        w3 = np.random.rand(1,)
        bias = np.random.rand(1,)

        for i in range(epochs):
            for j in range(len(inputs)):
                # Entrada (x1, x2, x3)
                x1, x2, x3 = inputs[j]

                # Cálculo da saída
                z = w1 * x1 + w2 * x2 + w3 * x3 + bias
                prediction = self.ativacao_funcao(z)

                # Atualização dos pesos
                error = outputs[j][0] - prediction
                w1 = w1 + learning_rate * error * x1
                w2 = w2 + learning_rate * error * x2
                w3 = w3 + learning_rate * error * x3
                bias = bias + learning_rate * error

        return w1, w2, w3, bias

    # Predição
    def predict(self, weights, x1, x2, x3):
        z = x1 * weights[0] + x2 * weights[1] + x3 * weights[2] + weights[3]
        return self.ativacao_funcao(z)


# Execução principal
if __name__ == "__main__":
    # Exemplo de entradas: [piso, sujeira, obstaculos]
    inputs = [
        [2, 2, 0],
        [1, 8, 2],
        [3, 5, 4],
        [2, 1, 1],
        [1, 9, 3],
        [3, 6, 0],
        [2, 3, 2],
        [1, 7, 1],
        [3, 4, 3],
        [2, 0, 0]
    ]

    # Saída exemplo: só "potência" (1, 2 ou 3)
    # Para simplificação, vamos transformar em binário (se alta potência = 1, caso contrário = 0)
    outputs = [[0], [1], [1], [0], [1], [1], [0], [1], [1], [0]]

    perceptron = Perceptron()

    # Treinamento
    weights = perceptron.train(inputs, outputs, learning_rate=0.1, epochs=100)

    # Teste da predição
    print("Predição para (piso=2, sujeira=4, obstaculo=1):", perceptron.predict(weights, 2, 4, 1))
    print("Predição para (piso=1, sujeira=9, obstaculo=2):", perceptron.predict(weights, 1, 9, 2))
    print("Predição para (piso=3, sujeira=0, obstaculo=0):", perceptron.predict(weights, 3, 0, 0))
