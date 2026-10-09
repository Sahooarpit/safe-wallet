import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def load_and_build_graph():
    print("Downloading and loading Kaggle dataset...")

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "sergionefedov/crypto-exchange-fraud-and-wash-trading-detection",
        "crypto_fraud_master.csv"
    )

    wallets_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "sergionefedov/crypto-exchange-fraud-and-wash-trading-detection",
        "wallets.csv"
    )

    sender_col = 'wallet_from'
    receiver_col = 'wallet_to'
    label_col = 'manipulation_type'

    unique_wallets = sorted(list(set(df[sender_col]).union(set(df[receiver_col]))))
    node_mapping = {wallet: idx for idx, wallet in enumerate(unique_wallets)}

    source_nodes = df[sender_col].map(node_mapping).values
    target_nodes = df[receiver_col].map(node_mapping).values
    edge_index = torch.tensor(np.vstack([source_nodes, target_nodes]), dtype=torch.long)
    num_nodes = len(unique_wallets)
    
    # Kept features after removing highly correlated ones
    from_stats = df.groupby(sender_col).agg(
        out_degree=('tx_id', 'count'),
        mean_notional_out=('notional_usd', 'mean'),
        mean_gas_out=('gas_fee_usd', 'mean'),
        mean_burst_5m=('tx_burst_count_5m', 'mean'),
        mean_counterparty_div=('counterparty_diversity', 'mean'),
        mean_notional_zscore=('notional_zscore_wallet', 'mean'),
        night_ratio=('is_night', 'mean')
    )
    

    to_stats = df.groupby(receiver_col).agg(
        in_degree=('tx_id', 'count'),
        mean_notional_in=('notional_usd', 'mean')
    )

    wallet_features = pd.DataFrame({'wallet': unique_wallets}).set_index('wallet')
    wallet_features = wallet_features.join(from_stats).join(to_stats).fillna(0)
    
    scaler = StandardScaler()
    x_numpy = scaler.fit_transform(wallet_features.values)
    x = torch.tensor(x_numpy, dtype=torch.float)

    y = torch.zeros(num_nodes, dtype=torch.long)
    wash_tx = df[df[label_col] == 'wash_trade']
    wash_wallets = set(wash_tx[sender_col]).union(set(wash_tx[receiver_col]))
    
    wash_indices = [node_mapping[w] for w in wash_wallets]
    y[wash_indices] = 1

    indices = torch.randperm(num_nodes)
    split = int(num_nodes * 0.8)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:split]] = True
    test_mask[indices[split:]] = True

    graph_data = Data(x=x, edge_index=edge_index, y=y)
    graph_data.train_mask = train_mask
    graph_data.test_mask = test_mask

    print(f"Graph Built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, {x.shape[1]} features")
    return graph_data, scaler
