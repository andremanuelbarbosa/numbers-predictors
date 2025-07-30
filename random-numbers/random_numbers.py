import math
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def build_dataset(dataset_size: int, sequence_size: int, min_number: int = 1, max_number: int = 12) -> pd.DataFrame:
    data: list[int] = []
    for i in range(0, dataset_size):
        data.append(random.randint(min_number, max_number))
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


def find_best_score(dataset_size: int, numbers_size: int) -> tuple[float, int]:
    best_score = 0
    best_sequence_size = numbers_size
    for sequence_size in range(numbers_size, int(math.pow(numbers_size, 2))):
        dataset = build_dataset(dataset_size, sequence_size, 1, numbers_size)
        linear_regression, score = train_dataset(dataset, 0.9)
        if score > best_score:
            best_score = score
            best_sequence_size = sequence_size
    return best_score, best_sequence_size


def random_numbers(dataset_size: int, numbers_size: int) -> None:
    score, sequence_size = find_best_score(dataset_size, numbers_size)
    print(f'Score for Dataset of size {dataset_size} and Sequence of size {sequence_size} was {score}')


random_numbers(1000, 12)
