import sys
import yaml
import torch
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.lstm import FlashFloodLSTM
from src.data.dataloader import create_dataloaders
from src.training.trainer import Trainer
from src.training.losses import get_loss_function


def main():
    with open('configs/lstm_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_X_path=config['data']['train_X'],
        train_y_path=config['data']['train_y'],
        val_X_path=config['data']['val_X'],
        val_y_path=config['data']['val_y'],
        test_X_path=config['data']['test_X'],
        test_y_path=config['data']['test_y'],
        batch_size=config['training']['batch_size'],
        subsample_ratio=config['training'].get('subsample_ratio', None)
    )

    print("Creating model...")
    model = FlashFloodLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    criterion = get_loss_function(
        config['training']['loss'],
        pos_weight=config['training'].get('pos_weight', None)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config['artifacts']['models_dir']
    )

    training_log = trainer.train(
        epochs=config['training']['epochs'],
        checkpoint_interval=config['training']['checkpoint_interval']
    )

    log_df = pd.DataFrame(training_log)
    log_path = Path(config['artifacts']['logs_dir']) / 'training_log.csv'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nTraining log saved to: {log_path}")

    config_path = Path(config['artifacts']['logs_dir']) / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
