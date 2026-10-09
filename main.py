import argparse
import time
import torch
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.jit._script')
from web3 import Web3
from data_processor import load_and_build_graph
from sybil_detection_gnn_model import SybilDetectionGNN
from train_model import GraphTrainer
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1. Configuration Constants
HIDDEN_CHANNELS = 64
EPOCHS = 100
LEARNING_RATE = 0.01
MODEL_PATH = "wash_trade_model.pth"
SCALER_PATH = "feature_scaler.pkl"
# Public Ethereum RPC (Replace with your Infura/Alchemy endpoint if rate limited)
ETH_RPC_URL = "https://ethereum-rpc.publicnode.com"

def train_job():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 2. Data Pipeline
    # Unpack the tuple since we modified data_processor.py to also return the scaler
    data, scaler = load_and_build_graph()
    data = data.to(device)

    # 3. Model Initialization
    num_node_features = data.x.shape[1]
    num_classes = 2  # Fraud vs Not Fraud

    model = SybilDetectionGNN(
        in_channels=num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    trainer = GraphTrainer(model, data, optimizer, criterion)

    # 4. Execution Loop
    logging.info("Starting Training...")
    for epoch in range(1, EPOCHS + 1):
        loss = trainer.train_epoch()

        if epoch % 10 == 0:
            test_f1 = trainer.evaluate()
            logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Macro F1: {test_f1:.4f}')

    # Save the model and scaler for production (real-time ETH)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")


def monitor_job():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the Scaler
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        logging.error("Scaler not found! Run with --train first.")
        return

    # 2. Load the Model
    # Important: The model needs the expected number of input features (18)
    # We retrieve it here from the saved scaler's feature array size
    num_node_features = scaler.n_features_in_
    model = SybilDetectionGNN(
        in_channels=num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=2
    ).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        logging.error("Model not found! Run with --train first.")
        return
    model.eval()  # Set to inference mode
    logging.info("Model and Scaler loaded successfully.")

    # 3. Connect to Ethereum Node
    w3 = Web3(Web3.HTTPProvider(ETH_RPC_URL))
    if not w3.is_connected():
        logging.error("Failed to connect to Ethereum Node.")
        return
    logging.info(f"Connected to Ethereum Node! Current Block: {w3.eth.block_number}")

    # For a real pipeline, we'd maintain an in-memory sliding window or Redis DB
    # of the features. To demonstrate inference on live data, we'll monitor live blocks,
    # collect addresses, mock the other 17 features dynamically based on transaction sizes,
    # and run the GNN on the new subgraph.
    
    logging.info("Starting real-time block monitoring...")
    
    # Keep track of the last block monitored
    last_block = w3.eth.block_number
    
    while True:
        try:
            current_block = w3.eth.block_number
            if current_block > last_block:
                logging.info(f"Processing new block: {current_block}")
                block = w3.eth.get_block(current_block, full_transactions=True)
                
                # Build subgraph of this block
                unique_wallets_set = set()
                edges_list = []
                
                for tx in block.transactions:
                    sender = tx.get('from')
                    receiver = tx.get('to')
                    if sender and receiver:
                        unique_wallets_set.add(sender)
                        unique_wallets_set.add(receiver)
                        edges_list.append((sender, receiver))
                        
                if not edges_list:
                    logging.info("No valid transactions found in block.")
                    last_block = current_block
                    continue
                
                unique_wallets = list(unique_wallets_set)
                node_to_idx = {addr: idx for idx, addr in enumerate(unique_wallets)}
                
                source_nodes = [node_to_idx[edge[0]] for edge in edges_list]
                target_nodes = [node_to_idx[edge[1]] for edge in edges_list]
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(device)
                
                # Mock features simulating real-world metric fetching (Redis/DB)
                # In production, fetch: tx_volume, degree, burst counts, from your DB here
                # We initialize raw static dummy features that match the expected 18-dim input
                raw_features = np.random.rand(len(unique_wallets), num_node_features)
                
                # Normalize features via the training stat scaler! This is crucial.
                scaled_features = scaler.transform(raw_features)
                x = torch.tensor(scaled_features, dtype=torch.float).to(device)
                
                with torch.no_grad():
                    log_probs = model(x, edge_index)
                    # Class 1 is 'wash_trade'
                    probabilities = torch.exp(log_probs[:, 1]) 
                    
                risk_scores = (probabilities * 100).round()
                
                # Alert any high-risk wallets (e.g. > 85%)
                high_risk = [(unique_wallets[i], int(risk_scores[i].item())) for i in range(len(unique_wallets)) if risk_scores[i] > 85]
                
                if high_risk:
                    logging.warning(f"⚠️ Wash Trade Detected in block {current_block}! {len(high_risk)} suspicious addresses:")
                    for addr, score in high_risk[:5]:  # limit to top 5
                        logging.warning(f"   -> {addr}: {score}% Risk")
                else:
                    logging.info(f"Block {current_block} clean. Max risk: {risk_scores.max().item()}%")
                
                last_block = current_block
            
            time.sleep(12)  # Avg block time in PoS is ~12s
            
        except Exception as e:
            logging.error(f"Error processing block: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Wash Trade Detector")
    parser.add_argument("--train", action="store_true", help="Train the model & export artifacts")
    parser.add_argument("--monitor", action="store_true", help="Run the real-time Ethereum node monitor")
    
    args = parser.parse_args()
    
    if args.train:
        train_job()
    elif args.monitor:
        monitor_job()
    else:
        print("Please specify --train or --monitor")

