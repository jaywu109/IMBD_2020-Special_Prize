def rmse(y, y_predict):
        e = 0
        for i in range(len(y)):
                e = e + (y[i] - y_predict[i]) ** 2
        e = e / len(y)
        e = e ** 0.5
        return e
