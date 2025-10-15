import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, matthews_corrcoef,
                             confusion_matrix)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Data Preprocessing
def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    return df, label_encoders

# Quantum Angle Encoding
def quantum_angle_encoding(x):
    return torch.sin(np.pi * x), torch.cos(np.pi * x)

# Simulated PQC Layer
class SimulatedPQC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimulatedPQC, self).__init__()
        self.fc = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x):
        sin_vals, cos_vals = quantum_angle_encoding(x)
        q_features = torch.cat((sin_vals, cos_vals), dim=1)
        return self.fc(q_features)


# CE-GRU Attention Module
class CEGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CEGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(weights * out, dim=1)
        return context

# Q-LexNet Model
class QLexNet(nn.Module):
    def __init__(self, input_dim, pqc_dim, lstm_dim, gru_dim, num_classes):
        super(QLexNet, self).__init__()
        self.pqc = SimulatedPQC(input_dim, pqc_dim)
        self.lstm = nn.LSTM(pqc_dim, lstm_dim, batch_first=True)
        self.ce_gru = CEGRU(lstm_dim, gru_dim)
        self.classifier = nn.Linear(gru_dim, num_classes)

    def forward(self, x):
        x = self.pqc(x)
        x = x.unsqueeze(1).repeat(1, 10, 1)
        lstm_out, _ = self.lstm(x)
        out = self.ce_gru(lstm_out)
        return self.classifier(out)

# Training Function
def train_model(model, data_loader, criterion, optimizer, device):
    model.train()
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluation Metrics
def evaluate_metrics(y_true, y_pred, average='macro'):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    specificity = TN / (TN + FP + 1e-10)

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average=average) * 100
    recall = recall_score(y_true, y_pred, average=average) * 100
    f1 = f1_score(y_true, y_pred, average=average) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"Precision     : {precision:.2f}%")
    print(f"Recall        : {recall:.2f}%")
    print(f"F1 Score      : {f1:.2f}%")
    print(f"Specificity   : {specificity:.2f}%")
    print(f"Kappa Score   : {kappa:.4f}")
    print(f"MCC           : {mcc:.4f}")

# Testing Function
def test_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        evaluate_metrics(y_test.cpu().numpy(), preds.cpu().numpy())

# Full Training & Evaluation Pipeline
def run_full_pipeline(X_train, y_train, X_test, y_test, input_dim, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QLexNet(input_dim=input_dim, pqc_dim=128, lstm_dim=128, gru_dim=64, num_classes=num_classes).to(device)

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    for epoch in range(100):
        train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/100 complete.")

    test_model(model, torch.tensor(X_test, dtype=torch.float32),
               torch.tensor(y_test, dtype=torch.long), device)

# Load CSV Dataset 
if __name__ == '__main__':
    df = pd.read_csv(r"path_of_the_csv_datset.csv")
    target_column = "target" #set the target label of the csv dataset

    X = df.drop(columns=[target_column])
    y = df[target_column]

    df_cleaned, encoders = preprocess_data(pd.concat([X, y], axis=1))
    X = df_cleaned.drop(columns=[target_column]).values
    y = df_cleaned[target_column].values.astype(int)

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    run_full_pipeline(X_train, y_train, X_test, y_test, input_dim=X.shape[1], num_classes=len(np.unique(y)))