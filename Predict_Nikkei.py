import yfinance as yf
import pandas as pd
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

# Download data
start_date = datetime.datetime(2004, 1, 1)
end_date = datetime.datetime.now()

fx_data = yf.download('JPY=X', start=start_date, end=end_date)
nikkei_data = yf.download('^N225', start=start_date, end=end_date)

data = pd.DataFrame({
    'JPY_USD': fx_data['Close'],
    'Nikkei225': nikkei_data['Close']
})

data.dropna(inplace=True)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(data['JPY_USD'].values.reshape(-1, 1))
Y_scaled = scaler_Y.fit_transform(data['Nikkei225'].values.reshape(-1, 1))

X = torch.tensor(X_scaled, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(Y_scaled, dtype=torch.float32).view(-1, 1)

# Create dataset
dataset = TensorDataset(X, Y)

# Split into train and test sets
train_size = len(dataset) // 2
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

# Define improved model
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super(ImprovedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = ImprovedNeuralNetwork()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2000
for epoch in range(epochs):
    for X_train, Y_train in train_loader:
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    for X_test, Y_test in test_loader:
        test_predictions = model(X_test)
        # Convert the test data to numpy for plotting
        X_test_np = X_test.numpy()
        Y_test_np = scaler_Y.inverse_transform(Y_test.numpy())
        test_predictions_np = scaler_Y.inverse_transform(test_predictions.numpy())

# Optional: Predicting for JPY/USD rate of 165
with torch.no_grad():
    input_value = torch.tensor(scaler_X.transform([[165]]), dtype=torch.float32).view(-1, 1)
    predicted_scaled = model(input_value)
    predicted_nikkei = scaler_Y.inverse_transform(predicted_scaled.numpy())
    print(f'Predicted Nikkei 225 at JPY/USD 165: {predicted_nikkei[0][0]:.2f}')

# Visualize with actual values
plt.scatter(X_test_np, Y_test_np, color='blue', label='Actual Test Data', alpha=0.6)
plt.scatter(X_test_np, test_predictions_np, color='red', label='Predicted Test Data', alpha=0.6)
# plt.plot(X_test_np, test_predictions_np, color='red', label='Fitted Line on Test Data')

plt.xlabel('JPY/USD Exchange Rate')
plt.ylabel('Nikkei 225 Index')
plt.legend()
plt.title('Nikkei 225 vs. JPY/USD Exchange Rate')
plt.show()

