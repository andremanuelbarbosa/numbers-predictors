import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def build_dataset(data_size: int, data_increment: int, sequence_size: int) -> pd.DataFrame:
    data: list[int] = []
    for i in range(0, data_size):
        data.append(i * data_increment)
    table = []
    indexes = []
    columns = []
    for i in range(sequence_size, 0, -1):
        columns.append(f'Y-{i}')
    columns.append('Y')
    for i in range(sequence_size, data_size):
        indexes.append(i - sequence_size + 1)
        table.append(data[i - sequence_size:i + 1])
    return pd.DataFrame(data=np.array(table), index=indexes, columns=columns)


def train_dataset(data: pd.DataFrame, train_size: float) -> tuple[LinearRegression, float]:
    y = data["Y"]
    x = data.drop("Y", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=1, shuffle=True)
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    score = linear_regression.score(x_test, y_test).round(3)
    return linear_regression, score


def predict_number(linear_regression: LinearRegression, sequence: list[int], expected: int) -> int:
    prediction_x = np.array(sequence).reshape(-1, len(sequence))
    prediction_y = round(linear_regression.predict(prediction_x)[0])
    print(f'Prediction for Sequence {sequence} is {prediction_y} / {expected}')
    return prediction_y


dataset = build_dataset(100, 1, 5)
lr, s = train_dataset(dataset, 0.8)
predict_number(lr, [95, 96, 97, 98, 99], 100)
predict_number(lr, [195, 196, 197, 198, 199], 200)
predict_number(lr, [495, 496, 497, 498, 499], 500)
predict_number(lr, [995, 996, 997, 998, 999], 1000)
