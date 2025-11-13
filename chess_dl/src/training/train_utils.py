import torch
import chess
import numpy as np
from typing import List, Tuple

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert a chess board to tensor representation."""
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    tensor = torch.zeros(12, 8, 8)
    
    for i, piece in enumerate(pieces):
        for pos in board.pieces(chess.Piece.from_symbol(piece).piece_type, 
                              chess.Piece.from_symbol(piece).color):
            rank, file = chess.square_rank(pos), chess.square_file(pos)
            tensor[i][rank][file] = 1
    
    return tensor.reshape(64, 12)  # Reshape to (64 squares, 12 piece channels)

def legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """Create a mask for legal moves."""
    mask = torch.zeros(4672)  # Maximum possible moves in chess
    for move in board.legal_moves:
        move_idx = move_to_index(move)
        mask[move_idx] = 1
    return mask

def move_to_index(move: chess.Move) -> int:
    """Convert a chess move to an index."""
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion if move.promotion else 0
    
    # Calculate unique index for the move
    index = from_square * 64 + to_square
    if promotion:
        index = 64 * 64 + (promotion - 2) * 64 * 64 + from_square * 64 + to_square
    
    return index

def index_to_move(index: int) -> chess.Move:
    """Convert an index back to a chess move."""
    if index >= 64 * 64:
        # Promotion move
        promotion_piece = ((index - 64 * 64) // (64 * 64)) + 2
        remaining_index = index % (64 * 64)
        from_square = remaining_index // 64
        to_square = remaining_index % 64
        return chess.Move(from_square, to_square, promotion_piece)
    else:
        # Regular move
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self.position = 0
        
    def push(self, state: torch.Tensor, policy: torch.Tensor, value: float):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, policy, value)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, policies, values = zip(*[self.buffer[i] for i in batch])
        return (torch.stack(states), 
                torch.stack(policies), 
                torch.tensor(values).float().unsqueeze(1))
    
    def __len__(self) -> int:
        return len(self.buffer)
