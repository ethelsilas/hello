import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your dataset (assuming df1 is your pandas DataFrame)
df1 = pd.read_csv(r'C:\Users\ESilas\Downloads\Apple_dataset.csv')

# Extract the time series data
data = df1['Close'].values.reshape(-1, 1)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create sequences for the LSTM model
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Define the sequence length (number of time steps to look back)
seq_length = 10

# Create sequences for training
sequences = create_sequences(data_scaled, seq_length)

# Split the data into training and testing sets
train_size = int(len(sequences) * 0.8)
train, test = sequences[:train_size], sequences[train_size:]

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# Reshape the input data for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Streamlit app
st.title("Stock Price Prediction App")

# Get user input for a specific date
selected_date = st.date_input("Select a date", min_value=df1.index.min().date(), max_value=df1.index.max().date())

# Convert selected date to the corresponding index in your DataFrame
selected_index = df1.index.get_loc(selected_date)

# Prepare the input data for prediction
input_data = data_scaled[selected_index - seq_length:selected_index].reshape(1, -1, 1)

# Make predictions
predicted_price = model.predict(input_data)

# Invert the scaling to get the actual stock price
predicted_price_inv = scaler.inverse_transform(predicted_price)

# Display the predicted stock price
st.write(f"Predicted Stock Price for {selected_date}: ${predicted_price_inv[0, 0]:.2f}")

# Plot the actual and predicted stock prices
st.line_chart(pd.DataFrame({
    'Actual': scaler.inverse_transform(data[selected_index - seq_length:selected_index + 1].reshape(-1, 1)).flatten(),
    'Predicted': np.concatenate([np.nan] * seq_length, predicted_price_inv.flatten())
}))
