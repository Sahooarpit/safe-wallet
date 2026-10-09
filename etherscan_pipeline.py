import argparse
import time
import torch
import numpy as np
import pandas as pd
import requests
import joblib
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.jit._script')

from sybil_detection_gnn_model import SybilDetectionGNN
from etherscan import Etherscan
from etherscan.utils.parsing import ResponseParser as response_parser

def __run_v2(func, api_key, net):
    def wrapper(*args, **kwargs):
        query_string = func(*args, **kwargs)
        url = f"https://api.etherscan.io/v2/api?chainid=1&{query_string}&apikey={api_key}"
        r = requests.get(url, headers={"User-Agent": ""})
        return response_parser.parse(r)
    return wrapper

Etherscan._Etherscan__run = __run_v2

import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)

import os
from dotenv import load_dotenv

load_dotenv()
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
MODEL_PATH = "wash_trade_model.pth"
SCALER_PATH = "feature_scaler.pkl"
HIDDEN_CHANNELS = 64

def load_model_and_scaler():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        logging.error("Scaler not found! Train the model first.")
        return None, None, None

    num_node_features = scaler.n_features_in_
    model = SybilDetectionGNN(
        in_channels=num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=2
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        logging.error("Model weights not found! Train the model first.")
        return None, None, None
        
    model.eval()
    return model, scaler, device

def get_most_recent_active_wallet():
    logging.info("No address provided. Fetching the most recent active wallet from the latest Ethereum block...")
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_getBlockByNumber&tag=latest&boolean=true&apikey={ETHERSCAN_API_KEY}"
    try:
        response = requests.get(url).json()
        if 'result' in response and response['result'] and 'transactions' in response['result']:
            txs = response['result']['transactions']
            if len(txs) > 0:
                addr = txs[0]['from']
                logging.info(f"Found active wallet: {addr}")
                return addr
    except Exception as e:
        logging.error(f"Failed to fetch latest block: {e}")
    return "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

def evaluate_address_history(target_address):
    model, scaler, device = load_model_and_scaler()
    if model is None:
        return

    logging.info(f"Pulling transaction history for {target_address} via etherscan-python...")
    eth = Etherscan(ETHERSCAN_API_KEY)
    
    try:
        txs = eth.get_normal_txs_by_address_paginated(
            address=target_address,
            page=1,
            offset=10000,
            startblock=0,
            endblock=99999999,
            sort='desc'
        )
    except Exception as e:
        logging.error(f"Etherscan API Error: {e}")
        return

    logging.info(f"Fetched {len(txs)} recent transactions.")
    if len(txs) == 0:
        return

    df = pd.DataFrame(txs)
    df = df.dropna(subset=['from', 'to'])
    df = df[df['to'] != '']
    df['value'] = (df['value'].astype(float) / 1e18) * 3000
    
    df['gasPrice'] = df['gasPrice'].astype(float)
    df['gasUsed'] = df['gasUsed'].astype(float)
    df['gas_fee_usd'] = (df['gasPrice'] * df['gasUsed'] / 1e18) * 3000
    
    df['timeStamp'] = df['timeStamp'].astype(int)
    df['datetime'] = pd.to_datetime(df['timeStamp'], unit='s')
    
    sender_col = 'from'
    receiver_col = 'to'

    # Compute calculable features
    df['is_night'] = ((df['datetime'].dt.hour >= 22) | (df['datetime'].dt.hour <= 4)).astype(int)
    df['notional_zscore_wallet'] = df.groupby(sender_col)['value'].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0).fillna(0)
    
    df = df.sort_values([sender_col, 'datetime'])
    df.set_index('datetime', inplace=True)
    df['tx_burst_count_5m'] = df.groupby(sender_col)['hash'].rolling('5min').count().reset_index(level=0, drop=True)
    df.reset_index(inplace=True)

    unique_wallets = sorted([w for w in set(df[sender_col]).union(set(df[receiver_col])) if w])
    node_mapping = {wallet: idx for idx, wallet in enumerate(unique_wallets)}

    source_nodes = df[sender_col].map(node_mapping).dropna().values
    target_nodes = df[receiver_col].map(node_mapping).dropna().values
    edge_index = torch.tensor(np.vstack([source_nodes, target_nodes]), dtype=torch.long).to(device)
    
    logging.info("Engineering features from on-chain data...")
    from_stats = df.groupby(sender_col).agg(
        out_degree=('hash', 'count'),
        mean_notional_out=('value', 'mean'),
        mean_gas_out=('gas_fee_usd', 'mean'),
        mean_burst_5m=('tx_burst_count_5m', 'mean'),
        mean_notional_zscore=('notional_zscore_wallet', 'mean'),
        night_ratio=('is_night', 'mean'),
        unique_counterparties=('to', 'nunique')
    )
    from_stats['mean_counterparty_div'] = from_stats['unique_counterparties'] / from_stats['out_degree']
    from_stats = from_stats.drop(columns=['unique_counterparties'])
    
    to_stats = df.groupby(receiver_col).agg(
        in_degree=('hash', 'count'),
        mean_notional_in=('value', 'mean')
    )

    wallet_features = pd.DataFrame({'wallet': unique_wallets}).set_index('wallet')
    wallet_features = wallet_features.join(from_stats).join(to_stats).fillna(0)
    
    # Ensure columns match data_processor.py exactly
    expected_cols = [
        'out_degree', 'mean_notional_out', 'mean_gas_out', 'mean_burst_5m', 
        'mean_counterparty_div', 'mean_notional_zscore', 'night_ratio', 
        'in_degree', 'mean_notional_in'
    ]
    wallet_features = wallet_features[expected_cols]

    scaled_features = scaler.transform(wallet_features.values)
    x = torch.tensor(scaled_features, dtype=torch.float).to(device)

    logging.info("Running evaluation through Graph Neural Network...")
    with torch.no_grad():
        log_probs = model(x, edge_index)
        probabilities = torch.exp(log_probs[:, 1]) 
        
    risk_scores = (probabilities * 100).round()
    
    results = []
    for i, wallet in enumerate(unique_wallets):
        results.append({"wallet": wallet, "risk_score": risk_scores[i].item()})
        
    results = sorted(results, key=lambda d: d['risk_score'], reverse=True)
    
    print(f"\n--- EVALUATION RESULTS FOR NETWORK OF {target_address} ---")
    for res in results[:10]:
        if res['risk_score'] > 50:
            print(f"??  HIGH RISK: {res['wallet']} -> {res['risk_score']}% probability of Wash Trading")
        else:
            print(f"?  CLEAN: {res['wallet']} -> {res['risk_score']}% probability of Wash Trading")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, required=False, help="Ethereum address to scan (defaults to latest active wallet)")
    args = parser.parse_args()
    
    logging.info("Starting scanning mode...")
    try:
        while True:
            target = args.address
            if not target:
                target = get_most_recent_active_wallet()
                
            if target:
                evaluate_address_history(target)
                
            time.sleep(1.0)
            if args.address:
                break
    except KeyboardInterrupt:
        logging.info("Scanning stopped by user.")





