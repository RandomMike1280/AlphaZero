import numpy as np
import chess
from .game import Game
from .move_encoding import action_int_to_uci, uci_to_action_int


class ChessGame(Game):
    """
    Implementation of Chess game for AlphaZero using python-chess.
    
    Board representation uses python-chess and is encoded for neural network input.
    """
    
    def __init__(self):
        """Initialize Chess game parameters."""
        self.row_count = 8
        self.column_count = 8
        self.action_size = 73 * 8 * 8  # From-square and to-square (includes illegal moves)
        
        # Piece type mapping
        self.piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
    def get_initial_state(self):
        """
        Returns the initial chess board.
        
        Returns:
            A python-chess Board object in the starting position.
        """
        return chess.Board()
    
    def get_next_state(self, state, action, player):
        """
        Makes a move on the chess board.
        
        Args:
            state: Current chess board state (python-chess Board).
            action: Integer representing a move (encoded via move_encoding.py).
            player: 1 for white, -1 for black (unused in this implementation).
            
        Returns:
            New board state after the move is made.
        """
        next_state = state.copy()
        move_uci = action_int_to_uci(action) # Gets base UCI like "h7h8" or "h7h8r"
        
        if move_uci is None:
            print(f"Warning: action_int_to_uci returned None for action {action}. Cannot make move.")
            return next_state
            
        # --- Promotion Logic moved here ---
        from_sq_name = move_uci[0:2]
        to_sq_name = move_uci[2:4]
        promotion_char = move_uci[4] if len(move_uci) == 5 else None
        
        from_square = chess.parse_square(from_sq_name)
        to_square = chess.parse_square(to_sq_name)
        
        # Check if it's a pawn promotion scenario based on state
        piece = next_state.piece_at(from_square)
        promotion_piece = None # Default no promotion
        if piece is not None and piece.piece_type == chess.PAWN:
            is_white_promo = (next_state.turn == chess.WHITE and chess.square_rank(to_square) == 7)
            is_black_promo = (next_state.turn == chess.BLACK and chess.square_rank(to_square) == 0)
            if is_white_promo or is_black_promo:
                # If UCI already specified an underpromotion, use it
                if promotion_char == 'n': promotion_piece = chess.KNIGHT
                elif promotion_char == 'b': promotion_piece = chess.BISHOP
                elif promotion_char == 'r': promotion_piece = chess.ROOK
                # Otherwise, default to QUEEN promotion
                else: promotion_piece = chess.QUEEN 
        
        # Create the move object with potential promotion
        move = chess.Move(from_square, to_square, promotion=promotion_piece)
        # --- End Promotion Logic ---

        # Make the move if legal
        if move in next_state.legal_moves:
            next_state.push(move)
        
        return next_state
    
    def __init__(self):
        """Initialize Chess game parameters."""
        self.row_count = 8
        self.column_count = 8
        self.action_size = 73 * 8 * 8
        
        # Piece type mapping
        self.piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
        # Move cache for faster repeated lookups
        self._move_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 10000  # Limit cache size to prevent memory issues
    
    def get_valid_moves(self, state):
        """
        Returns a binary vector of valid moves.
        
        Args:
            state: Current chess board state (python-chess Board).
            
        Returns:
            Binary vector of valid moves with length action_size.
        """
        # Check if position is in cache
        cache_key = state.fen()
        if cache_key in self._move_cache:
            self._cache_hits += 1
            return self._move_cache[cache_key].copy()  # Return a copy to prevent modification
        
        self._cache_misses += 1
        
        # Pre-allocate array with zeros
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        
        # Early return if no legal moves
        if not state.legal_moves:
            # Cache the result before returning
            if len(self._move_cache) < self._max_cache_size:
                self._move_cache[cache_key] = valid_moves.copy()
            return valid_moves
        
        # Process moves in batches for better performance
        valid_indices = []
        
        # Direct conversion from Move objects to action integers
        for move in state.legal_moves:
            # Convert to UCI format only once
            uci = move.uci()
            
            # Use the existing function with optimized input
            action = uci_to_action_int(uci)
            if action is not None:
                valid_indices.append(action)
        
        # Use numpy's advanced indexing to set multiple values at once - more efficient
        if valid_indices:
            valid_moves[valid_indices] = 1
        
        # Cache the result before returning
        if len(self._move_cache) < self._max_cache_size:
            self._move_cache[cache_key] = valid_moves.copy()
        elif len(self._move_cache) >= self._max_cache_size:
            # If cache is full, clear it (simple strategy)
            # A more sophisticated approach would use LRU cache
            if self._cache_hits / (self._cache_hits + self._cache_misses) < 0.5:
                self._move_cache = {}
                self._cache_hits = 0
                self._cache_misses = 0
        
        return valid_moves
    
    def check_win(self, state, action):
        """
        Checks if the last action led to a win (checkmate or stalemate).
        
        Args:
            state: Current chess board state (python-chess Board).
            action: Last action taken.
            
        Returns:
            True if the last action led to checkmate or stalemate, False otherwise.
        """
        # Checkmate or stalemate ends the game
        return state.is_checkmate() or state.is_stalemate()
    
    def get_value_and_terminated(self, state, action):
        """
        Checks if the game is over and returns the value.
        
        Args:
            state: Current chess board state (python-chess Board).
            action: Last action taken.
            
        Returns:
            (value, terminated): 
                - value: 1 if win for the last player, 0 if draw, -1 if loss
                - terminated: True if the game is over, False otherwise
        """
        if action is None:
            return 0, False
        
        # Game is over if checkmate, stalemate, insufficient material, etc.
        if state.is_game_over():
            if state.is_checkmate():
                # The player who just moved wins
                return 1, True
            else:
                # Draw (stalemate, insufficient material, 50-move rule, etc.)
                return 0, True
                
        return 0, False
    
    def change_perspective(self, state, player):
        """
        Changes the perspective of the state to the given player.
        
        Args:
            state: Current chess board state (python-chess Board).
            player: Player to change perspective to (1 for white, -1 for black).
            
        Returns:
            Board from the perspective of the given player.
        """
        # In chess, we don't modify the board, but rather check if the turn matches the player
        perspective_state = state.copy()
        
        # If player is black (-1) and it's white's turn, or vice versa, flip the turn
        if (player == -1 and perspective_state.turn == chess.WHITE) or (player == 1 and perspective_state.turn == chess.BLACK):
            perspective_state = perspective_state.mirror()
            # Note: mirror() flips the board, but python-chess handles the turn automatically
            
        return perspective_state
    
    def get_encoded_state(self, state):
        """
        Encodes the chess board for neural network input.
        
        This uses the encoding from AlphaZero paper: 12 planes for piece positions 
        (6 for white pieces, 6 for black pieces), plus additional features like 
        castling rights, en passant, and turn.
        
        Args:
            state: Current chess board state (python-chess Board).
            
        Returns:
            Encoded state with shape (19, 8, 8):
                - 6 planes for white pieces (pawn, knight, bishop, rook, queen, king)
                - 6 planes for black pieces
                - 4 planes for castling rights
                - 1 plane for en passant
                - 1 plane for move count
                - 1 plane for current turn
        """
        encoded_state = np.zeros((19, self.row_count, self.column_count), dtype=np.float32)
        
        # Add piece planes
        for i, piece_type in enumerate(self.piece_types):
            # White pieces (channels 0-5)
            for square in state.pieces(piece_type, chess.WHITE):
                row, col = self._square_to_position(square)
                encoded_state[i][row][col] = 1
                
            # Black pieces (channels 6-11)
            for square in state.pieces(piece_type, chess.BLACK):
                row, col = self._square_to_position(square)
                encoded_state[i + 6][row][col] = 1
                
        # Castling rights
        if state.has_kingside_castling_rights(chess.WHITE):
            encoded_state[12].fill(1)
        if state.has_queenside_castling_rights(chess.WHITE):
            encoded_state[13].fill(1)
        if state.has_kingside_castling_rights(chess.BLACK):
            encoded_state[14].fill(1)
        if state.has_queenside_castling_rights(chess.BLACK):
            encoded_state[15].fill(1)
            
        # En passant
        if state.ep_square is not None:
            row, col = self._square_to_position(state.ep_square)
            encoded_state[16][row][col] = 1
            
        # Move count (50-move rule counter)
        encoded_state[17].fill(state.halfmove_clock / 100.0)  # Normalize
        
        # Turn (0 for black, 1 for white)
        if state.turn == chess.WHITE:
            encoded_state[18].fill(1)
            
        return encoded_state
    
    def _square_to_position(self, square):
        """
        Convert a chess.square to a (row, col) position.
        
        Args:
            square: A chess square (0-63).
            
        Returns:
            Tuple of (row, col).
        """
        row = 7 - (square // 8)  # Invert row because chess board has row 0 at the bottom
        col = square % 8
        return row, col
    
    def get_opponent(self, player):
        """
        Returns the opponent of the given player.
        
        Args:
            player: Current player (1 for white, -1 for black).
            
        Returns:
            Opponent player (-1 for black, 1 for white).
        """
        return -player
    
    def get_move_from_action(self, state, action):
        """
        Convert an action index to a chess move.
        
        Args:
            state: Current chess board state.
            action: Action index.
            
        Returns:
            A chess.Move object.
        """
        from_square = action // 64
        to_square = action % 64
        
        # Check if we need to specify promotion
        promotion = None
        piece = state.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            if (to_square < 8 and state.turn == chess.BLACK) or (to_square >= 56 and state.turn == chess.WHITE):
                promotion = chess.QUEEN
                
        return chess.Move(from_square, to_square, promotion)