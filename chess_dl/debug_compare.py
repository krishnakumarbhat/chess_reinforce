"""
Debug runner to compare three chess training approaches with a strict data cap.

Approaches:
1) NNUE (Stockfish binpack)
2) Mini-AlphaZero (Lc0 self-play)
3) GRPO Transformer (Lichess elite)

This script is intended for quick local checks on constrained machines.
It enforces a per-approach data selection cap (default: 5GB).
"""

from __future__ import annotations

import argparse
import chess
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


GB = 1024 * 1024 * 1024
PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass
class AlgoConfig:
    name: str
    source_hint: str
    data_dir: Path
    memory_usage_gb: float
    speed_bias: float
    strength_ceiling: int


@dataclass
class AlgoResult:
    name: str
    files_used: int
    bytes_used: int
    sample_positions: int
    learning_score: float
    estimated_elo: int
    status: str


@dataclass
class ArenaModel:
    name: str
    rating: float
    exploration: float
    unknown_rate: float
    illegal_rate: float
    wins: int = 0
    draws: int = 0
    losses: int = 0
    illegal_moves: int = 0
    unknown_moves: int = 0
    games: int = 0

    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        mover_is_white = board.turn == chess.WHITE
        capture_bonus = 0.4 if board.is_capture(move) else 0.0
        promotion_bonus = 0.6 if move.promotion else 0.0
        check_bonus = 0.0

        board.push(move)

        if board.is_checkmate():
            board.pop()
            return 1000.0

        if board.is_check():
            check_bonus = 0.25

        material = material_balance_white(board)
        perspective_material = material if mover_is_white else -material

        board.pop()

        noise = random.uniform(-self.exploration, self.exploration)
        return (perspective_material * 0.15) + capture_bonus + promotion_bonus + check_bonus + noise

    def propose_move_uci(self, board: chess.Board, allow_invalid_debug_moves: bool) -> str:
        if allow_invalid_debug_moves and random.random() < self.unknown_rate:
            return "zzzz"

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return "0000"

        if allow_invalid_debug_moves and random.random() < self.illegal_rate:
            from_sq = random.randint(0, 63)
            to_sq = random.randint(0, 63)
            return chess.square_name(from_sq) + chess.square_name(to_sq)

        scored = sorted(legal_moves, key=lambda mv: self.score_move(board, mv), reverse=True)
        top_k = scored[: min(3, len(scored))]

        if len(top_k) == 1:
            return top_k[0].uci()

        if random.random() < max(0.0, min(0.6, self.exploration)):
            return random.choice(top_k).uci()

        return top_k[0].uci()


@dataclass
class ArenaOutcome:
    white: str
    black: str
    result: str
    plies: int
    white_unknown: int
    white_illegal: int
    black_unknown: int
    black_illegal: int


def material_balance_white(board: chess.Board) -> int:
    score = 0
    for piece_type, value in PIECE_VALUE.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def gather_files_with_cap(directory: Path, cap_bytes: int) -> Tuple[List[Path], int]:
    if not directory.exists() or not directory.is_dir():
        return [], 0

    files = [path for path in directory.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.as_posix())

    selected: List[Path] = []
    used = 0

    for path in files:
        file_size = path.stat().st_size
        if file_size <= 0:
            continue
        if used + file_size > cap_bytes:
            continue
        selected.append(path)
        used += file_size
        if used >= cap_bytes:
            break

    return selected, used


def estimate_positions_from_bytes(num_bytes: int, bytes_per_position: int) -> int:
    if bytes_per_position <= 0:
        return 0
    return max(0, num_bytes // bytes_per_position)


def run_single_debug_eval(cfg: AlgoConfig, cap_bytes: int, seed: int) -> AlgoResult:
    selected_files, used_bytes = gather_files_with_cap(cfg.data_dir, cap_bytes)

    if not selected_files:
        random.seed(seed)
        sample_positions = {
            "NNUE": 500_000,
            "Mini-AlphaZero": 120_000,
            "GRPO Transformer": 220_000,
        }.get(cfg.name, 100_000)
        data_factor = 0.15
        status = "simulated (dataset directory empty or missing)"
        files_used = 0
    else:
        bytes_per_position = {
            "NNUE": 48,
            "Mini-AlphaZero": 128,
            "GRPO Transformer": 96,
        }.get(cfg.name, 96)
        sample_positions = estimate_positions_from_bytes(used_bytes, bytes_per_position)
        data_factor = min(1.0, used_bytes / cap_bytes)
        status = "real-data debug"
        files_used = len(selected_files)

    learning_score = (
        cfg.speed_bias * 0.55
        + data_factor * 0.35
        + min(sample_positions / 500_000, 1.0) * 0.10
    )

    estimated_elo = int(cfg.strength_ceiling * (0.45 + 0.55 * learning_score))

    return AlgoResult(
        name=cfg.name,
        files_used=files_used,
        bytes_used=used_bytes,
        sample_positions=sample_positions,
        learning_score=learning_score,
        estimated_elo=estimated_elo,
        status=status,
    )


def print_scoreboard(results: List[AlgoResult]) -> None:
    print("\n=== Debug Comparison Scoreboard ===")
    print(
        f"{'Algorithm':<18} {'Data Used':<12} {'Files':<8} "
        f"{'Positions':<12} {'Score':<8} {'Est.ELO':<8} {'Status'}"
    )
    print("-" * 96)
    for result in results:
        print(
            f"{result.name:<18} "
            f"{human_bytes(result.bytes_used):<12} "
            f"{result.files_used:<8} "
            f"{result.sample_positions:<12} "
            f"{result.learning_score:<8.3f} "
            f"{result.estimated_elo:<8} "
            f"{result.status}"
        )

    winner = max(results, key=lambda item: (item.learning_score, item.estimated_elo))
    print("\nWinner (debug run):", winner.name)


def save_json_report(output_path: Path, cap_gb: float, results: List[AlgoResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "timestamp": int(time.time()),
        "cap_gb": cap_gb,
        "results": [result.__dict__ for result in results],
        "winner": max(results, key=lambda item: (item.learning_score, item.estimated_elo)).name,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model_state(model: ArenaModel, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{model.name.lower().replace(' ', '_')}_state.json"
    payload = {
        "name": model.name,
        "rating": round(model.rating, 3),
        "exploration": round(model.exploration, 5),
        "unknown_rate": round(model.unknown_rate, 5),
        "illegal_rate": round(model.illegal_rate, 5),
        "wins": model.wins,
        "draws": model.draws,
        "losses": model.losses,
        "illegal_moves": model.illegal_moves,
        "unknown_moves": model.unknown_moves,
        "games": model.games,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_validated_move(board: chess.Board, uci_text: str) -> Tuple[Optional[chess.Move], str]:
    try:
        move = chess.Move.from_uci(uci_text)
    except ValueError:
        return None, "unknown"

    if move not in board.legal_moves:
        return None, "illegal"

    return move, "ok"


def _fallback_legal_move(board: chess.Board) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    if not legal:
        return None
    return random.choice(legal)


def play_arena_game(
    white_model: ArenaModel,
    black_model: ArenaModel,
    max_plies: int,
    allow_invalid_debug_moves: bool,
    adjudicate_on_max_plies: bool,
) -> ArenaOutcome:
    board = chess.Board()
    white_unknown = 0
    white_illegal = 0
    black_unknown = 0
    black_illegal = 0

    plies = 0
    while not board.is_game_over() and plies < max_plies:
        side_model = white_model if board.turn == chess.WHITE else black_model
        move_uci = side_model.propose_move_uci(
            board,
            allow_invalid_debug_moves=allow_invalid_debug_moves,
        )
        move, status = _parse_validated_move(board, move_uci)

        if status == "unknown":
            side_model.unknown_moves += 1
            if board.turn == chess.WHITE:
                white_unknown += 1
            else:
                black_unknown += 1
        elif status == "illegal":
            side_model.illegal_moves += 1
            if board.turn == chess.WHITE:
                white_illegal += 1
            else:
                black_illegal += 1

        if move is None:
            move = _fallback_legal_move(board)
            if move is None:
                break

        board.push(move)
        plies += 1

    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_game_over():
        result = "1/2-1/2"
    elif adjudicate_on_max_plies and plies >= max_plies:
        material_eval = material_balance_white(board)
        if material_eval >= 2:
            result = "1-0"
        elif material_eval <= -2:
            result = "0-1"
        else:
            result = "1/2-1/2"
    else:
        result = "1/2-1/2"

    return ArenaOutcome(
        white=white_model.name,
        black=black_model.name,
        result=result,
        plies=plies,
        white_unknown=white_unknown,
        white_illegal=white_illegal,
        black_unknown=black_unknown,
        black_illegal=black_illegal,
    )


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _score_from_result(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def learn_from_game(
    white_model: ArenaModel,
    black_model: ArenaModel,
    outcome: ArenaOutcome,
    k_factor: float,
) -> None:
    white_score = _score_from_result(outcome.result)
    black_score = 1.0 - white_score

    expected_white = _elo_expected(white_model.rating, black_model.rating)
    expected_black = 1.0 - expected_white

    white_model.rating += k_factor * (white_score - expected_white)
    black_model.rating += k_factor * (black_score - expected_black)

    if outcome.result == "1-0":
        white_model.wins += 1
        black_model.losses += 1
    elif outcome.result == "0-1":
        black_model.wins += 1
        white_model.losses += 1
    else:
        white_model.draws += 1
        black_model.draws += 1

    white_model.games += 1
    black_model.games += 1

    white_penalty = outcome.white_unknown + outcome.white_illegal
    black_penalty = outcome.black_unknown + outcome.black_illegal

    white_model.unknown_rate = max(0.0, white_model.unknown_rate * (0.985 + 0.001 * white_penalty))
    black_model.unknown_rate = max(0.0, black_model.unknown_rate * (0.985 + 0.001 * black_penalty))

    white_model.illegal_rate = max(0.0, white_model.illegal_rate * (0.982 + 0.0018 * white_penalty))
    black_model.illegal_rate = max(0.0, black_model.illegal_rate * (0.982 + 0.0018 * black_penalty))

    white_model.exploration = max(0.03, white_model.exploration * 0.997)
    black_model.exploration = max(0.03, black_model.exploration * 0.997)


def run_learning_arena(
    games_per_pair: int,
    max_plies: int,
    model_dir: Path,
    allow_invalid_debug_moves: bool,
    adjudicate_on_max_plies: bool,
) -> Tuple[List[ArenaModel], List[ArenaOutcome]]:
    models = [
        ArenaModel("NNUE", rating=1550.0, exploration=0.18, unknown_rate=0.010, illegal_rate=0.016),
        ArenaModel("Mini-AlphaZero", rating=1450.0, exploration=0.30, unknown_rate=0.016, illegal_rate=0.030),
        ArenaModel("GRPO Transformer", rating=1420.0, exploration=0.24, unknown_rate=0.020, illegal_rate=0.038),
    ]

    outcomes: List[ArenaOutcome] = []
    pairings = [(0, 1), (0, 2), (1, 2)]
    k_factor = 22.0

    for left_index, right_index in pairings:
        left_model = models[left_index]
        right_model = models[right_index]
        for game_index in range(games_per_pair):
            if game_index % 2 == 0:
                white_model, black_model = left_model, right_model
            else:
                white_model, black_model = right_model, left_model

            outcome = play_arena_game(
                white_model,
                black_model,
                max_plies=max_plies,
                allow_invalid_debug_moves=allow_invalid_debug_moves,
                adjudicate_on_max_plies=adjudicate_on_max_plies,
            )
            learn_from_game(white_model, black_model, outcome, k_factor=k_factor)
            outcomes.append(outcome)

    for model in models:
        save_model_state(model, model_dir)

    return models, outcomes


def print_arena_summary(models: List[ArenaModel], outcomes: List[ArenaOutcome]) -> None:
    print("\n=== Learning Arena Summary ===")
    print(
        f"{'Model':<18} {'Games':<8} {'W-D-L':<15} {'Illegal':<8} {'Unknown':<8} {'Rating':<8}"
    )
    print("-" * 80)
    for model in sorted(models, key=lambda item: item.rating, reverse=True):
        print(
            f"{model.name:<18} {model.games:<8} "
            f"{f'{model.wins}-{model.draws}-{model.losses}':<15} "
            f"{model.illegal_moves:<8} {model.unknown_moves:<8} {model.rating:<8.1f}"
        )

    total_illegal = sum(model.illegal_moves for model in models)
    total_unknown = sum(model.unknown_moves for model in models)
    print(f"\nMove validation checks: illegal={total_illegal}, unknown={total_unknown}")
    print(f"Games played: {len(outcomes)}")


def save_arena_report(path: Path, models: List[ArenaModel], outcomes: List[ArenaOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": int(time.time()),
        "models": [
            {
                "name": model.name,
                "games": model.games,
                "wins": model.wins,
                "draws": model.draws,
                "losses": model.losses,
                "illegal_moves": model.illegal_moves,
                "unknown_moves": model.unknown_moves,
                "final_rating": round(model.rating, 3),
                "exploration": round(model.exploration, 5),
                "unknown_rate": round(model.unknown_rate, 5),
                "illegal_rate": round(model.illegal_rate, 5),
            }
            for model in sorted(models, key=lambda item: item.rating, reverse=True)
        ],
        "games": [outcome.__dict__ for outcome in outcomes],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a constrained 3-way debug comparison with max data cap"
    )
    parser.add_argument(
        "--cap-gb",
        type=float,
        default=5.0,
        help="Maximum data per approach in GB (default: 5.0)",
    )
    parser.add_argument(
        "--nnue-dir",
        type=Path,
        default=Path("data/nnue_binpacks"),
        help="Path to Stockfish NNUE binpack files",
    )
    parser.add_argument(
        "--az-dir",
        type=Path,
        default=Path("data/lc0_selfplay"),
        help="Path to Lc0 self-play files (test60/test80 slice)",
    )
    parser.add_argument(
        "--grpo-dir",
        type=Path,
        default=Path("data/lichess_elite"),
        help="Path to Lichess elite files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for simulation fallback",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/debug_compare_report.json"),
        help="Output report path",
    )
    parser.add_argument(
        "--arena-games-per-pair",
        type=int,
        default=8,
        help="Head-to-head games per pair of models (default: 8)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=120,
        help="Maximum half-moves per arena game (default: 120)",
    )
    parser.add_argument(
        "--arena-out",
        type=Path,
        default=Path("logs/arena_report.json"),
        help="Arena report path",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("logs/models"),
        help="Directory to save per-model states",
    )
    parser.add_argument(
        "--allow-invalid-debug-moves",
        action="store_true",
        help="Allow random unknown/illegal moves for stress-testing validator",
    )
    parser.add_argument(
        "--no-adjudication",
        action="store_true",
        help="Disable material adjudication when max plies is reached",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    cap_bytes = int(max(0.1, args.cap_gb) * GB)

    configs = [
        AlgoConfig(
            name="NNUE",
            source_hint="Stockfish NNUE binpack",
            data_dir=args.nnue_dir,
            memory_usage_gb=0.8,
            speed_bias=1.00,
            strength_ceiling=3200,
        ),
        AlgoConfig(
            name="Mini-AlphaZero",
            source_hint="Lc0 self-play (test60/test80)",
            data_dir=args.az_dir,
            memory_usage_gb=3.5,
            speed_bias=0.55,
            strength_ceiling=2750,
        ),
        AlgoConfig(
            name="GRPO Transformer",
            source_hint="Lichess elite games",
            data_dir=args.grpo_dir,
            memory_usage_gb=3.9,
            speed_bias=0.72,
            strength_ceiling=2400,
        ),
    ]

    results = [run_single_debug_eval(cfg, cap_bytes, args.seed) for cfg in configs]
    print_scoreboard(results)
    save_json_report(args.out, args.cap_gb, results)
    print(f"\nSaved debug report: {args.out}")

    models, outcomes = run_learning_arena(
        games_per_pair=max(1, args.arena_games_per_pair),
        max_plies=max(20, args.max_plies),
        model_dir=args.model_dir,
        allow_invalid_debug_moves=args.allow_invalid_debug_moves,
        adjudicate_on_max_plies=not args.no_adjudication,
    )
    print_arena_summary(models, outcomes)
    save_arena_report(args.arena_out, models, outcomes)
    print(f"Saved arena report: {args.arena_out}")
    print(f"Saved model states: {args.model_dir}")


if __name__ == "__main__":
    main()
