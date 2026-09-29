import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def load_and_build_graph():
    print("Downloading and loading Kaggle dataset...")

    # 1. Load the datasets directly into Pandas DataFrames
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

    print("Dataset Loaded Successfully. First 5 records:")
    print(df.head())

    # ==========================================
    # Column Mapping
    # ==========================================
    sender_col = 'wallet_from'
    receiver_col = 'wallet_to'
    label_col = 'manipulation_type'  # Target to detect 'wash_trade'

    print("\nMapping tabular data to graph structure...")

    # 2. Map string addresses to integer Node IDs
    unique_wallets = sorted(list(set(df[sender_col]).union(set(df[receiver_col]))))
    node_mapping = {wallet: idx for idx, wallet in enumerate(unique_wallets)}

    source_nodes = df[sender_col].map(node_mapping).values
    target_nodes = df[receiver_col].map(node_mapping).values

    # 3. Build edge_index (The graph connections)
    edge_index = torch.tensor(np.vstack([source_nodes, target_nodes]), dtype=torch.long)

    # 4. Define Node Features (x)
    num_nodes = len(unique_wallets)
    
    # Feature engineering! Aggregating node features from master table
    from_stats = df.groupby(sender_col).agg(
        out_degree=('tx_id', 'count'),
        mean_notional_out=('notional_usd', 'mean'),
        sum_notional_out=('notional_usd', 'sum'),
        max_notional_out=('notional_usd', 'max'),
        mean_gas_out=('gas_fee_usd', 'mean'),
        mean_time_since_prev=('time_since_prev_tx_sec', 'mean'),
        mean_burst_5m=('tx_burst_count_5m', 'mean'),
        mean_counterparty_div=('counterparty_diversity', 'mean'),
        mean_counterparty_reuse=('counterparty_reuse_ratio', 'mean'),
        mean_round_trip=('round_trip_score', 'mean'),
        mean_pair_txn_count=('pair_txn_count', 'mean'),
        mean_vol_zscore=('token_volume_zscore_1h', 'mean'),
        mean_price_dev=('price_deviation_vwap_pct', 'mean'),
        mean_notional_zscore=('notional_zscore_wallet', 'mean'),
        dex_ratio=('is_dex', 'mean'),
        night_ratio=('is_night', 'mean'),
        major_ratio=('token_is_major', 'mean'),
        mean_token_liq=('token_liquidity_score', 'mean')
    )
    
    to_stats = df.groupby(receiver_col).agg(
        in_degree=('tx_id', 'count'),
        mean_notional_in=('notional_usd', 'mean'),
        sum_notional_in=('notional_usd', 'sum'),
        max_notional_in=('notional_usd', 'max'),
    )

    wallet_features = pd.DataFrame({'wallet': unique_wallets}).set_index('wallet')
    
    # Add wallet age from wallets.csv
    if 'wallet_age_days' in wallets_df.columns:
        w_age = wallets_df.set_index('wallet')['wallet_age_days']
        wallet_features['wallet_age_days'] = wallet_features.index.map(w_age).fillna(0)

    # Combine all stats and fill NaN with 0
    wallet_features = wallet_features.join(from_stats).join(to_stats).fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    x_numpy = scaler.fit_transform(wallet_features.values)
    x = torch.tensor(x_numpy, dtype=torch.float)

    # 5. Define Node Labels (y) - We are detecting wash trading
    y = torch.zeros(num_nodes, dtype=torch.long)

    # Find nodes involved in wash trading (either sending or receiving)
    wash_tx = df[df[label_col] == 'wash_trade']
    wash_wallets = set(wash_tx[sender_col]).union(set(wash_tx[receiver_col]))
    
    wash_indices = [node_mapping[w] for w in wash_wallets]
    y[wash_indices] = 1

    # 6. Create Train/Test Masks (80/20 split)
    indices = torch.randperm(num_nodes)
    split = int(num_nodes * 0.8)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:split]] = True
    test_mask[indices[split:]] = True

    # 7. Construct PyG Data Object
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    graph_data.train_mask = train_mask
    graph_data.test_mask = test_mask

    print(f"Graph Built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, {x.shape[1]} features")
    print(f"Wash Trading Nodes (Class 1): {y.sum().item()}, Normal Nodes (Class 0): {len(y) - y.sum().item()}")
    return graph_data
