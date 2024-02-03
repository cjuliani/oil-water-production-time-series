import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

matplotlib.use('TkAgg')


def draw_well(df, name):
    """Displays time series"""
    plt.figure(dpi=100)
    for feat in ['BHP', 'WHP', 'CHOKE_1_PERC', 'CHOKE_2_PERC', 'OIL_PROD', 'GAS_PROD', 'WAT_PROD', 'WAT_INJ']:
        plt.plot(df[feat], label=feat)
    plt.yscale('log')
    plt.xticks(np.arange(df.DATE.shape[0])[::100], labels=df.DATE[::100], rotation=45, fontsize=6)
    plt.grid()
    plt.xlabel('Date')
    plt.legend(fontsize=5)
    plt.title('Well %s'%name)
    plt.show()


def check_feature_correlations(x):
    """Visualize correlation matrix using a heatmap."""
    # Calculate correlation matrix
    correlation_matrix = x.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()


def get_histogram(x, bins):
    plt.hist(x, bins=bins, color='black')
    plt.show()


def get_scatterplot(features, x, y):
    sns.scatterplot(x=x, y=y, data=features)
    plt.grid(True, linestyle='--', color='gray')
    plt.show()


def create_sequences(data, prod, seq_length, nodata=1.):
    """Create sequences of time series data."""
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = prod[i + seq_length]
        tmp = data[i:i + seq_length]
        cond = list(tmp.sum() == seq_length)
        if (label == nodata) or (True in cond):
            continue
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def build_model(sequence_length, features_n, lr=1e-3):
    # Build the GRU model
    custom_optimizer = Adam(learning_rate=lr)

    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=(sequence_length, features_n)))
    model.add(Dense(1))
    model.compile(
        optimizer=custom_optimizer,
        loss='mean_squared_error',
        metrics=['mae', 'mape'])
    return model


def balance_and_randomize(X, y, sequence_length, features_n):
    """For data balancing purposes, splits data within different
    groups, e.g. by interval of OP of 0.05, then estimate the
    maximum number of data features N among the interval groups.
    We later resample N data for each interval group. This is to
    avoid preferential sampling and biased training."""
    intervals = np.arange(0,1.05,0.05).tolist()
    bins = pd.cut(y.reshape((-1,)), bins=intervals, include_lowest=True)
    max_n = bins.value_counts().max()

    bins = np.digitize(y.reshape((-1,)), intervals)
    indices_by_interval = {interval: [] for interval in intervals}
    for i, bin in enumerate(bins):
        interval = intervals[bin - 1]  # adjust index since np.digitize is 1-indexed
        indices_by_interval[interval].append(i)

    # Resample data based on interval groups (data balancing)
    X_resampled, y_resampled = [], []
    for interval, indices in indices_by_interval.items():
        if not indices:
            continue
        rand_indices = np.random.choice(indices, size=max_n, replace=True)
        X_resampled.append(X[rand_indices])
        y_resampled.append(y[rand_indices])

    X_resampled = np.array(X_resampled).reshape((-1, sequence_length, features_n))
    y_resampled = np.array(y_resampled).reshape((-1, 1))

    # Randomize data sequences
    rand = np.random.choice(range(len(X)), len(y))
    X_balanced, y_balanced = X_resampled[rand], y_resampled[rand]

    return X_balanced, y_balanced
