import argparse
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.jit._script')
import torch
import joblib

from data_processor import load_and_build_graph
from sybil_detection_gnn_model import SybilDetectionGNN
from train_model import GraphTrainer
from etherscan_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration Constants
HIDDEN_CHANNELS = 64
EPOCHS = 100
LEARNING_RATE = 0.01
MODEL_PATH = "wash_trade_model.pth"
SCALER_PATH = "feature_scaler.pkl"


def train_job():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Data Pipeline
    data, scaler = load_and_build_graph()
    data = data.to(device)

    # 2. Model Initialization
    num_node_features = data.x.shape[1]
    num_classes = 2  # Fraud vs Normal

    model = SybilDetectionGNN(
        in_channels=num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    trainer = GraphTrainer(model, data, optimizer, criterion)

    # 3. Execution Loop
    logging.info("Starting Training...")
    for epoch in range(1, EPOCHS + 1):
        loss = trainer.train_epoch()

        if epoch % 10 == 0:
            test_f1 = trainer.evaluate()
            logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Macro F1: {test_f1:.4f}')

    # Save model weights and scaler
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")


def monitor_job(target_address=None, continuous=False):
    """
    Delegates real-time monitoring and inference to the Etherscan pipeline.
    Pulls live transactions, calculates graph features, and runs inference.
    """
    run_pipeline(target_address=target_address, continuous=continuous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Wash Trading Detector CLI")
    parser.add_argument("--train", action="store_true", help="Train the GNN model & export artifacts")
    parser.add_argument("--monitor", action="store_true", help="Run the real-time Etherscan pipeline monitor")
    parser.add_argument("--address", type=str, default=None, help="Target Ethereum address to evaluate (optional)")
    parser.add_argument("--continuous", action="store_true", help="Continuously poll and scan in real time")

    args = parser.parse_args()

    if args.train:
        train_job()
    elif args.monitor:
        monitor_job(target_address=args.address, continuous=args.continuous)
    else:
        parser.print_help()
