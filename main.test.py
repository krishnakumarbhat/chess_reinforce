def test_init_standard_starting_position():
    # Create a ChessEnv instance
    env = ChessEnv()
    
    # Check if the board is initialized with standard starting position
    assert env.board.fen().split(' ')[0] == chess.STARTING_FEN.split(' ')[0]
    
    # Verify action space is correct
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 4672
    
    # Verify observation space is correct
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (8, 8, 12)
    assert env.observation_space.low.all() == 0
    assert env.observation_space.high.all() == 1