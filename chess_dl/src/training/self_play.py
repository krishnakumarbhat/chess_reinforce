import torch
import chess
import numpy as np
from typing import Tuple, List
from ..models.chess_transformer import ChessTransformer
from .train_utils import board_to_tensor, legal_moves_mask, move_to_index, index_to_move

class SelfPlayAgent:
    def __init__(self, model: ChessTransformer, device: torch.device, temperature: float = 1.0):
        self.model = model
        self.device = device
        self.temperature = temperature
        
    def select_move(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """Select a move using the model and return the move and its value."""
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None, 0.0
            
        # Create mask for legal moves
        mask = torch.zeros(4672).to(self.device)
        for move in legal_moves:
            mask[move_to_index(move)] = 1
            
        with torch.no_grad():
            policy_logits, value = self.model(state)
            policy_logits = policy_logits.squeeze(0)
            
            # Apply mask and temperature
            policy_logits = policy_logits * mask  # Zero out illegal moves
            policy_logits = policy_logits / self.temperature
            
            # Handle case where all moves are masked
            if torch.max(policy_logits) == float('-inf') or torch.max(policy_logits) == 0:
                # If no legal moves found in policy, choose randomly from legal moves
                move = np.random.choice(legal_moves)
                return move, value.item()
            
            policy_probs = torch.softmax(policy_logits, dim=0)
            
            # Get only legal moves probabilities
            legal_moves_indices = [move_to_index(move) for move in legal_moves]
            legal_probs = policy_probs[legal_moves_indices]
            
            # Normalize probabilities
            legal_probs = legal_probs / legal_probs.sum()
            
            # Sample move from legal moves
            try:
                move_idx = np.random.choice(len(legal_moves), p=legal_probs.cpu().numpy())
                move = legal_moves[move_idx]
            except:
                # Fallback to random legal move if sampling fails
                move = np.random.choice(legal_moves)
            
        return move, value.item()

def play_game(agent: SelfPlayAgent) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Play a complete game of chess using self-play."""
    board = chess.Board()
    game_history = []
    
    while not board.is_game_over():
        state = board_to_tensor(board)
        move, value = agent.select_move(board)
        
        if move is None:  # No legal moves available
            break
            
        # Store state and move
        policy = torch.zeros(4672)
        policy[move_to_index(move)] = 1
        game_history.append((state, policy, value))
        
        # Make move
        board.push(move)
        
        # Break if game is too long
        if len(game_history) > 200:  # Prevent infinite games
            break
    
    # Get game result
    if board.is_checkmate():
        final_value = -1 if board.turn else 1
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        final_value = 0
    else:
        final_value = 0
    
    # Update values based on game result
    for i in range(len(game_history)):
        state, policy, _ = game_history[i]
        adjusted_value = final_value * (-1 if i % 2 == 1 else 1)
        game_history[i] = (state, policy, adjusted_value)
    
    return game_history

def self_play_iteration(model: ChessTransformer, 
                       device: torch.device,
                       num_games: int = 100,
                       temperature: float = 1.0) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Perform one iteration of self-play."""
    agent = SelfPlayAgent(model, device, temperature)
    all_game_history = []
    
    for _ in range(num_games):
        game_history = play_game(agent)
        all_game_history.extend(game_history)
    
    return all_game_history
