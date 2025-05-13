import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from scipy.special import logsumexp

DATA_DIR = 'UCI_HAR_Dataset'

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

def train_hmms(X_train, y_train, n_states=4, cov_type='diag'):
    models = {}
    for act in np.unique(y_train):
        model = GaussianHMM(n_components=n_states,
                            covariance_type=cov_type,
                            n_iter=100, verbose=False)
        model.fit(X_train[y_train == act])
        models[act] = model
        print(f"[HMM] Trained for '{act}' ({np.sum(y_train==act)} windows)")
    return models

def build_transition_matrix(y_train, labels):
    M = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    transitions = np.ones((M, M))  # smoothing Laplace = 1
    for (a, b) in zip(y_train[:-1], y_train[1:]):
        transitions[label_to_index[a], label_to_index[b]] += 1
    transitions /= transitions.sum(axis=1, keepdims=True)
    return transitions

def particle_filter(obs_seq, models, trans_matrix, N=500):
    labels = list(models.keys())
    M = len(labels)
    T = obs_seq.shape[0]

    label_to_index = {label: i for i, label in enumerate(labels)}

    particles = np.random.choice(M, size=N)
    weights = np.ones(N) / N
    estimates = []

    for t in range(T):
        obs = obs_seq[t].reshape(1, -1)
        
        # Cập nhật trọng số
        log_weights = np.zeros(N)
        for i in range(N):
            act = labels[particles[i]]
            log_weights[i] = models[act].score(obs)
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)

        # Dự đoán
        cnt = np.zeros(M)
        for i, p in enumerate(particles):
            cnt[p] += weights[i]
        estimates.append(labels[np.argmax(cnt)])

        # Resample particles có chuyển trạng thái
        idx = np.random.choice(N, size=N, p=weights)
        particles = particles[idx]

        # Áp dụng state transition
        for i in range(N):
            current_state = particles[i]
            particles[i] = np.random.choice(
                M, p=trans_matrix[current_state]
            )

        weights = np.ones(N) / N

    return np.array(estimates)

def main():
    print("Loading features...")
    X_train, y_train = load_features('train')
    X_test, y_test = load_features('test')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = train_hmms(X_train, y_train, n_states=6)

    labels = np.unique(y_train)
    trans_matrix = build_transition_matrix(y_train, labels)

    print("Running Particle Filter...")
    estimates = particle_filter(X_test, models, trans_matrix, N=500)

    accuracy = np.mean(estimates == y_test)
    print(f"Overall accuracy (HMM+PF): {accuracy:.3f}")

    plt.figure(figsize=(12,4))
    plt.plot(y_test[:200], label='True', marker='o')
    plt.plot(estimates[:200], label='Estimates', marker='x', alpha=0.7)
    plt.xticks(rotation=45)
    plt.xlabel('Window index')
    plt.ylabel('Activity label')
    plt.title('True vs Estimated Activity (first 200 windows)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
