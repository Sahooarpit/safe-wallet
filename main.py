import torch
from data_processor import load_and_build_graph
from sybil_detection_gnn_model import SybilDetectionGNN
from train_model import GraphTrainer


def main():
    # 1. Configuration
    hidden_channels = 64
    epochs = 100
    learning_rate = 0.01

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data Pipeline
    data = load_and_build_graph().to(device)

    # 3. Model Initialization
    num_node_features = data.x.shape[1]
    num_classes = 2  # Fraud vs Not Fraud

    model = SybilDetectionGNN(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Use weights in CrossEntropy to handle class imbalance if needed
    criterion = torch.nn.NLLLoss()

    trainer = GraphTrainer(model, data, optimizer, criterion)

    # 4. Execution Loop
    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch()

        if epoch % 10 == 0:
            test_f1 = trainer.evaluate()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Macro F1: {test_f1:.4f}')


if __name__ == "__main__":
    main()
