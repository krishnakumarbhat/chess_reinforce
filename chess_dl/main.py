"""
Chess AI System using Transformer Architecture and Reinforcement Learning
Main entry point for training and evaluation
"""

import os
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import chess
import logging
import json
from tqdm import tqdm

from src.models.chess_transformer import create_chess_transformer
from src.training.train_utils import ReplayBuffer
from src.training.self_play import self_play_iteration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

# Create project directory structure
def create_project_structure():
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "models/transformer",
        "models/rl",
        "models/checkpoints",
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "configs",
        "notebooks",
        "tests",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        Path(directory) / '.gitkeep'

def setup_training():
    # Initialize model, optimizer, and other components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = create_chess_transformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    return model, optimizer, device

def train_on_dataset(model, optimizer, device, num_epochs=10):
    logging.info("Starting initial training on dataset")
    metrics = {
        'train_losses': [],
        'validation_scores': [],
        'epochs': []
    }
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Generate training data through self-play
        logging.info(f"Epoch {epoch+1}: Generating self-play games...")
        game_histories = self_play_iteration(model, device, num_games=10)
        
        # Add games to replay buffer
        for state, policy, value in game_histories:
            replay_buffer.push(state, policy, value)
        
        # Training loop
        logging.info(f"Epoch {epoch+1}: Training on collected games...")
        num_batches = min(len(replay_buffer) // 32, 1000)  # Limit number of batches per epoch
        
        for _ in tqdm(range(num_batches)):
            states, policies, values = replay_buffer.sample(32)
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            
            optimizer.zero_grad()
            policy_output, value_output = model(states)
            
            # Calculate losses
            policy_loss = nn.CrossEntropyLoss()(policy_output, policies)
            value_loss = nn.MSELoss()(value_output, values)
            total_loss = policy_loss + value_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        avg_epoch_loss = epoch_loss / num_batches
        metrics['train_losses'].append(avg_epoch_loss)
        metrics['epochs'].append(epoch)
        
        logging.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, metrics)
            plot_metrics(metrics)
    
    return metrics

def self_play_improvement(model, optimizer, device, num_games=1000):
    logging.info("Starting self-play improvement phase")
    self_play_metrics = {
        'game_outcomes': [],
        'model_improvements': [],
        'game_lengths': []
    }
    
    # Initialize replay buffer for self-play
    replay_buffer = ReplayBuffer(capacity=100000)
    
    for game_batch in tqdm(range(0, num_games, 10)):
        # Generate games through self-play
        game_histories = self_play_iteration(model, device, num_games=10)
        
        # Track metrics
        for state, policy, value in game_histories:
            replay_buffer.push(state, policy, value)
        
        # Training on collected games
        model.train()
        batch_loss = 0.0
        num_batches = 100  # Number of training batches per self-play batch
        
        for _ in range(num_batches):
            states, policies, values = replay_buffer.sample(32)
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            
            optimizer.zero_grad()
            policy_output, value_output = model(states)
            
            policy_loss = nn.CrossEntropyLoss()(policy_output, policies)
            value_loss = nn.MSELoss()(value_output, values)
            total_loss = policy_loss + value_loss
            
            total_loss.backward()
            optimizer.step()
            
            batch_loss += total_loss.item()
        
        avg_batch_loss = batch_loss / num_batches
        self_play_metrics['model_improvements'].append(avg_batch_loss)
        
        if (game_batch + 10) % 100 == 0:
            save_checkpoint(model, optimizer, game_batch, self_play_metrics, prefix="self_play")
            plot_self_play_metrics(self_play_metrics)
    
    return self_play_metrics

def save_checkpoint(model, optimizer, epoch, metrics, prefix="training"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"models/checkpoints/{prefix}_checkpoint_{timestamp}.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    
    # Save metrics separately for analysis
    metrics_path = f"logs/{prefix}_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    logging.info(f"Saved checkpoint and metrics to {checkpoint_path} and {metrics_path}")

def plot_metrics(metrics):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epochs'], metrics['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epochs'], metrics['validation_scores'] if 'validation_scores' in metrics else metrics['train_losses'])
    plt.title('Validation Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'logs/training_plot_{timestamp}.png')
    plt.close()

def plot_self_play_metrics(metrics):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['game_outcomes'] if 'game_outcomes' in metrics else [])
    plt.title('Game Outcomes')
    plt.xlabel('Game')
    plt.ylabel('Outcome')
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics['model_improvements'])
    plt.title('Model Improvements')
    plt.xlabel('Game Batch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(metrics['game_lengths'] if 'game_lengths' in metrics else [])
    plt.title('Game Lengths')
    plt.xlabel('Game')
    plt.ylabel('Moves')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'logs/self_play_plot_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    create_project_structure()
    
    # Setup training components
    model, optimizer, device = setup_training()
    
    # Initial training on dataset
    training_metrics = train_on_dataset(model, optimizer, device, num_epochs=50)
    
    # Self-play improvement
    self_play_metrics = self_play_improvement(model, optimizer, device, num_games=1000)
    
    logging.info("Training completed successfully!")
