import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def build_dataset(dataset_size: int, sequence_size: int, data_increment: int = 1) -> pd.DataFrame:
    data: list[int] = []
    for i in range(0, dataset_size):
        data.append(i * data_increment)
    table = []
    indexes = []
    columns = []
    for i in range(sequence_size, 0, -1):
        columns.append(f'Y-{i}')
    columns.append('Y')
    for i in range(sequence_size, dataset_size):
        indexes.append(i - sequence_size + 1)
        table.append(data[i - sequence_size:i + 1])
    return pd.DataFrame(data=np.array(table), index=indexes, columns=columns)


def train_dataset(data: pd.DataFrame, train_size: float) -> tuple[LinearRegression, float]:
    y = data["Y"]
    x = data.drop("Y", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=1, shuffle=True)
    linear_regression = LinearRegression()
    linear_regression.fit(x_train.values, y_train)
    score = linear_regression.score(x_test.values, y_test).round(3)
    return linear_regression, score


def predict_number(linear_regression: LinearRegression, sequence_size: int, start_number: int) -> int:
    sequence = []
    for i in range(start_number, start_number + sequence_size):
        sequence.append(i)
    prediction_x = np.array(sequence).reshape(-1, len(sequence))
    return round(linear_regression.predict(prediction_x)[0])


def incremental_sequence(dataset_size: int, sequence_size: int) -> None:
    dataset = build_dataset(dataset_size, sequence_size)
    linear_regression, score = train_dataset(dataset, 0.8)
    print(f'Score for Dataset of size {dataset_size} and Sequence of size {sequence_size} was {score}')
    for i in range(0, 20):
        expected_number = 100 * int(math.pow(10, i))
        predicted_number = predict_number(linear_regression, sequence_size, expected_number - sequence_size)
        if expected_number != predicted_number:
            print(f'Predicted number was {predicted_number} but expected was {expected_number}')


incremental_sequence(100, 5)
