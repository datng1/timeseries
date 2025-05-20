import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DATA_DIR = 'UCI_HAR_Dataset'  # chỉnh sửa đường dẫn theo tên folder vừa giải nén


activity_labels = pd.read_csv(
    os.path.join(DATA_DIR, 'activity_labels.txt'),
    sep=' ', header=None, names=['id', 'activity']
)
label_map = dict(zip(activity_labels.id, activity_labels.activity))

def load_features(set_name):
    folder = os.path.join(DATA_DIR, set_name, 'Inertial Signals')
    acc_x = pd.read_csv(os.path.join(folder, f'body_acc_x_{set_name}.txt'),
                        delim_whitespace=True, header=None).values
    acc_y = pd.read_csv(os.path.join(folder, f'body_acc_y_{set_name}.txt'),
                        delim_whitespace=True, header=None).values
    acc_z = pd.read_csv(os.path.join(folder, f'body_acc_z_{set_name}.txt'),
                        delim_whitespace=True, header=None).values

    X = np.column_stack([
        acc_x.mean(axis=1), acc_y.mean(axis=1), acc_z.mean(axis=1),
        acc_x.std(axis=1),  acc_y.std(axis=1),  acc_z.std(axis=1)
    ])

    y_ids = pd.read_csv(
        os.path.join(DATA_DIR, set_name, f'y_{set_name}.txt'),
        header=None, names=['activity_id']
    )['activity_id'].values
    y = np.array([label_map[i] for i in y_ids])
    return X, y

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return x.detach().numpy()

def main():
    X_train, y_train = load_features('train')
    X_test, y_test = load_features('test')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    transformer = SimpleTransformer(input_dim=X_train.shape[1])

    with torch.no_grad():
        embedding_train = transformer(torch.tensor(X_train, dtype=torch.float32))
        embedding_test = transformer(torch.tensor(X_test, dtype=torch.float32))

    gmm = GaussianMixture(n_components=len(np.unique(y_train)), covariance_type='full', random_state=42)
    gmm.fit(embedding_train)

    predicted_labels = gmm.predict(embedding_test)

    label_encoder = {label: idx for idx, label in enumerate(np.unique(y_train))}
    y_test_encoded = np.array([label_encoder[label] for label in y_test])

    accuracy = np.mean(predicted_labels == y_test_encoded)
    print(f"Accuracy (Transformer + GMM): {accuracy:.3f}")

    plt.figure(figsize=(12,4))
    plt.plot(y_test_encoded[:200], label='True', marker='o')
    plt.plot(predicted_labels[:200], label='Predicted', marker='x', alpha=0.7)
    plt.xticks(rotation=45)
    plt.xlabel('Window index')
    plt.ylabel('Activity label (encoded)')
    plt.title('True vs Predicted Activity (Transformer + GMM)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()