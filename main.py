import pandas as pd
import tensorflow as tf

from utils import create_sequences, build_model, balance_and_randomize, get_histogram
from utils import check_feature_correlations, get_scatterplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

tf.keras.backend.set_epsilon(1)  # important for stable 'mape' metric
pd.options.mode.chained_assignment = None

# Load dataset and format it according to well type. We assign
# water injection features to each oil production well
data = pd.read_csv('data/prod_data_train.csv')
data_op_wells = data[data['WELL_TYPE']=='OP']  # oil production
data_op_wells = data_op_wells.reset_index().drop(columns=['WAT_INJ'])
data_wi_wells = data[data['WELL_TYPE']=='WI']  # water injection

op_well_group = []
for grp1 in data_op_wells.groupby(['WELLNAME']):
    # Format current OP well
    g1 = grp1[1]
    g1['DATE'] = pd.to_datetime(g1['DATE'].values.reshape((-1,)).tolist())

    merged_df = g1
    for grp2 in data_wi_wells.groupby(['WELLNAME']):
        # Add water injection data to current OP well
        g2 = grp2[1][['DATE', 'WAT_INJ']]
        g2['DATE'] = pd.to_datetime(g2['DATE'].values.reshape((-1,)).tolist())
        merged_df = pd.merge(merged_df, g2, on='DATE')

    op_well_group.append(merged_df)

# Define targets and related features
train_data = pd.concat(op_well_group, axis=0)
oil_prod = train_data['OIL_PROD'].values.reshape(-1, 1)
gas_prod = train_data['GAS_PROD'].values.reshape(-1, 1)
water_prod = train_data['WAT_PROD'].values.reshape(-1, 1)
features = train_data[['BHP', 'WHP', 'CHOKE_1_PERC', 'CHOKE_2_PERC', 'WAT_INJ_x', 'WAT_INJ_y']]

# Check outliers
get_histogram(oil_prod, bins=200)
get_histogram(water_prod, bins=200)
get_scatterplot(features, x='BHP', y='WHP')

# Check variable correlations and drop highly correlated ones
check_feature_correlations(features)
data_op_wells = data_op_wells.reset_index().drop(columns=['CHOKE_2_PERC'])
features_n = features.shape[1]


# Define sequence length and split the data into training and
# testing sets, i.e. sequences of data values (time series) later
# used as input to a GRU learning model
sequence_length = 10
X_op, y_op = create_sequences(features, oil_prod, sequence_length, nodata=1.)
X_wp, y_wp = create_sequences(features, water_prod, sequence_length, nodata=1.)

# --- OP: train and evaluate model
scaler = MinMaxScaler()
X_op_norm = scaler.fit_transform(X_op.reshape((-1, features_n))).reshape((-1, sequence_length, features_n))
y_op_norm = scaler.fit_transform(y_op)

X_balanced, y_balanced = balance_and_randomize(X_op_norm, y_op_norm, sequence_length, features_n)

split = int(0.8 * len(X_balanced))
X_train, X_test = X_balanced[:split], X_balanced[split:]
y_train, y_test = y_balanced[:split], y_balanced[split:]

model_op = build_model(sequence_length, features_n, 5e-4)
model_op.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)
loss, mae, mape = model_op.evaluate(X_test, y_test)
print(f'OP --- Test Loss: {loss}, MAE: {mae}, MAPE: {mape}')

# Make predictions on the test set, and get the true values at
# original scale
predictions = model_op.predict(X_test)
predictions_original_scale = scaler.inverse_transform(predictions)
y_test_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))
score_op = mean_absolute_percentage_error(y_test_original_scale, predictions_original_scale)

# --- WP: train and evaluate model
scaler = MinMaxScaler()
X_wp_norm = scaler.fit_transform(X_wp.reshape((-1, features_n))).reshape((-1, sequence_length, features_n))
y_wp_norm = scaler.fit_transform(y_wp)
X_balanced, y_balanced = balance_and_randomize(X_wp_norm, y_wp_norm, sequence_length, features_n)

split = int(0.8 * len(X_balanced))
X_train, X_test = X_balanced[:split], X_balanced[split:]
y_train, y_test = y_balanced[:split], y_balanced[split:]

model_wp = build_model(sequence_length, features_n, 5e-4)
model_wp.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)
loss, mae, mape = model_wp.evaluate(X_test, y_test)
print(f'WP --- Test Loss: {loss}, MAE: {mae}, MAPE: {mape}')

# Make predictions on the test set, and get the true values at
# original scale
predictions = model_op.predict(X_test)
predictions_original_scale = scaler.inverse_transform(predictions)
y_test_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))
score_wp = mean_absolute_percentage_error(y_test_original_scale, predictions_original_scale)

# --- Total score
metric = 0.8 * score_op + 0.2 * score_wp
print('Total score:', metric)
