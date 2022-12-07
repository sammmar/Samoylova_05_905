class SGD:
    def __init__(self, learning_rate, net, loss):
        """
        :param learning_rate: значение коэффициента скорости обучения
        :param net: объект нейронной сети
        :param loss: объект целевой функции
        """
        self.learning_rate = learning_rate
        self.net = net
        self.loss = loss

    def update_rule(self, dW):
        """
        прописать правило обновления обучаемых параметров
        :param dW: градиент обучаемых параметров
        """

    def minimize(self, dz_dl=1):
        """
        вычисление градиентов начиная с целевой функции, затем с конца по всем слоям нейронной сети
        :param dz_dl: нужно только для случая, если значение целевой функции не скаляр
        """
        dz_dl = self.loss.backward(dz_dl)
        for layer in reversed(self.net.parameters):
            dz_dl = layer.backward(dz_dl)
            layer.update_weights()
        return input
