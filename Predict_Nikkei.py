import yfinance as yf
import pandas as pd
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

# Step 1: Downloading the data from Yahoo Finance for JPY/USD, Nikkei 225, S&P 500, and Oil prices
print("Downloading data...")
start_date = datetime.datetime(2004, 1, 1)
end_date = datetime.datetime.now()

fx_data = yf.download('JPY=X', start=start_date, end=end_date)
nikkei_data = yf.download('^N225', start=start_date, end=end_date)
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
oil_data = yf.download('CL=F', start=start_date, end=end_date)

# Checking if the data was downloaded successfully
print(f"FX data: {fx_data.shape}, Nikkei data: {nikkei_data.shape}, S&P 500 data: {sp500_data.shape}, Oil data: {oil_data.shape}")

# Step 2: Creating a DataFrame and aligning all datasets by their dates, adding lagged Nikkei features
data = pd.DataFrame({
    'JPY_USD': fx_data['Close'],
    'Nikkei225': nikkei_data['Close'],
    'SP500': sp500_data['Close'],
    'Oil': oil_data['Close']
})

# Creating lagged Nikkei values (e.g., 1-day lag and 2-day lag)
data['Nikkei_Lag_1'] = data['Nikkei225'].shift(1)  # Previous day's Nikkei
data['Nikkei_Lag_2'] = data['Nikkei225'].shift(2)  # 2 days ago's Nikkei

# Drop rows with NaN values that result from shifting
data.dropna(inplace=True)
print(f"Data after adding lagged Nikkei values and dropping NaN values: {data.shape}")

# Step 3: Scaling the data using MinMaxScaler (Normalization to [0,1] range)
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

# Scale all input features (JPY/USD, SP500, Oil, and Lagged Nikkei values)
X_scaled = scaler_X.fit_transform(data[['JPY_USD', 'SP500', 'Oil', 'Nikkei_Lag_1', 'Nikkei_Lag_2']].values)

# Scale the target (Nikkei 225)
Y_scaled = scaler_Y.fit_transform(data['Nikkei225'].values.reshape(-1, 1))

# Convert to torch tensors for PyTorch model compatibility
X = torch.tensor(X_scaled, dtype=torch.float32)
Y = torch.tensor(Y_scaled, dtype=torch.float32).view(-1, 1)

# Debug: Checking data shapes after scaling
print(f"X_scaled shape: {X.shape}, Y_scaled shape: {Y.shape}")

# Step 4: Creating dataset and splitting it into train and test sets
dataset = TensorDataset(X, Y)
train_size = len(dataset) // 2
test_size = len(dataset) - train_size

# Split dataset into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

# Debug: Display the sizes of training and test datasets
print(f"Training set size: {train_size}, Test set size: {test_size}")

# Step 5: Defining a Neural Network with 2 hidden layers
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # Input to hidden layer with 32 neurons
        self.fc2 = nn.Linear(32, 16)         # Hidden layer with 16 neurons
        self.fc3 = nn.Linear(16, 1)          # Output layer
        self.dropout = nn.Dropout(0.2)       # Dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))          # ReLU activation for the first layer
        x = self.dropout(x)                  # Dropout after first layer
        x = torch.relu(self.fc2(x))          # ReLU activation for the second layer
        x = self.dropout(x)                  # Dropout after second layer
        x = self.fc3(x)                      # Output layer (no activation)
        return x

# Instantiate the model with the appropriate input dimension
model = ImprovedNeuralNetwork(input_dim=X.shape[1])  # Input dimension based on the number of features

# Step 6: Define the loss function and optimizer
criterion = nn.MSELoss()   # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Step 7: Training the model
epochs = 2000  # Number of epochs to train
print("Training the model...")
for epoch in range(epochs):
    for X_train, Y_train in train_loader:
        # Forward pass
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Debug: Print loss every 100 epochs to monitor training progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 8: Evaluating the model on the test set
print("Evaluating the model on test data...")
model.eval()
with torch.no_grad():
    for X_test, Y_test in test_loader:
        test_predictions = model(X_test)

        # Convert the test data back to original values
        X_test_np = scaler_X.inverse_transform(X_test.numpy())   # Inverse scaling of X_test
        Y_test_np = scaler_Y.inverse_transform(Y_test.numpy())   # Inverse scaling of actual Y_test
        test_predictions_np = scaler_Y.inverse_transform(test_predictions.numpy())  # Inverse scaling of predicted values

# Debug: Check some test results
print("Displaying some test predictions vs actual values:")
print(f"JPY/USD: {X_test_np[:5, 0]}, Actual Nikkei: {Y_test_np[:5].flatten()}, Predicted Nikkei: {test_predictions_np[:5].flatten()}")

# Step 9: Predicting Nikkei 225 value for specific values
with torch.no_grad():
    # Assume JPY/USD = 150, S&P 500 = 5800, Oil = 75, and use the most recent lagged Nikkei values
    latest_nikkei_lags = data.iloc[-1][['Nikkei_Lag_1', 'Nikkei_Lag_2']].values
    custom_input_values = [150, 5800, 75] + list(latest_nikkei_lags)  # Manually set JPY/USD, SP500, and Oil
    
    # Scale the custom input values
    input_values_scaled = scaler_X.transform([custom_input_values])
    input_tensor = torch.tensor(input_values_scaled, dtype=torch.float32).view(-1, X.shape[1])
    
    # Make prediction
    predicted_scaled = model(input_tensor)
    predicted_nikkei = scaler_Y.inverse_transform(predicted_scaled.numpy())
    
    print(f'Predicted Nikkei 225 at JPY/USD 150, S&P 500 5800, Oil (CL=F) 75: {predicted_nikkei[0][0]:.2f}')

# Step 10: Visualize the results
plt.scatter(X_test_np[:, 0], Y_test_np, color='blue', label='Actual Test Data', alpha=0.6)  # JPY/USD
plt.scatter(X_test_np[:, 0], test_predictions_np, color='red', label='Predicted Test Data', alpha=0.6)

plt.xlabel('JPY/USD Exchange Rate')
plt.ylabel('Nikkei 225 Index')
plt.legend()
plt.title('Nikkei 225 vs. JPY/USD Exchange Rate (Test Data)')
plt.show()
