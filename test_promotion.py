import chess
from alphazero.games.chess_game import ChessGame
from alphazero.games.move_encoding import uci_to_action_int, action_int_to_uci

def test_promotion_scenario():
    """
    Tests encoding/decoding and execution of pawn promotion moves.
    """
    game = ChessGame()
    # FEN: White pawn h7, White King a1, Black King a2, White to move
    fen = '8/7P/8/8/8/k7/8/K7 w - - 0 1'
    board = chess.Board(fen)

    print(f"Initial Board (FEN: {fen}):")
    print(board)
    print("-" * 20)

    # --- Test Queen Promotion --- 
    uci_q = "h7h8q"
    print(f"Testing Queen Promotion: {uci_q}")
    action_q = uci_to_action_int(uci_q)
    print(f"  Encoded Action: {action_q}")

    if action_q is not None:
        decoded_q = action_int_to_uci(action_q)
        print(f"  Decoded UCI: {decoded_q}")
        print(f"  Match: {decoded_q == uci_q}")

        # Simulate the move using the action integer
        next_board_q = game.get_next_state(board.copy(), action_q, 1) # Player 1 (White)
        print("  Board after move:")
        print(next_board_q)
        promoted_piece_q = next_board_q.piece_at(chess.H8)
        print(f"  Piece at h8: {promoted_piece_q}")
        if promoted_piece_q:
            print(f"  Is Queen: {promoted_piece_q.symbol() == 'Q'}")
        else:
            print("  Error: No piece found at h8")
    else:
        print("  Encoding failed.")
    print("-" * 20)

    # --- Test Rook Underpromotion --- 
    uci_r = "h7h8r"
    print(f"Testing Rook Underpromotion: {uci_r}")
    action_r = uci_to_action_int(uci_r)
    print(f"  Encoded Action: {action_r}")

    if action_r is not None:
        decoded_r = action_int_to_uci(action_r)
        print(f"  Decoded UCI: {decoded_r}")
        print(f"  Match: {decoded_r == uci_r}")

        # Simulate the move using the action integer
        next_board_r = game.get_next_state(board.copy(), action_r, 1) # Player 1 (White)
        print("  Board after move:")
        print(next_board_r)
        promoted_piece_r = next_board_r.piece_at(chess.H8)
        print(f"  Piece at h8: {promoted_piece_r}")
        if promoted_piece_r:
            print(f"  Is Rook: {promoted_piece_r.symbol() == 'R'}")
        else:
            print("  Error: No piece found at h8")
    else:
        print("  Encoding failed.")
    print("-" * 20)

    # --- Test Ambiguous Promotion (No piece specified) --- 
    uci_amb = "h7h8"
    print(f"Testing Ambiguous Promotion: {uci_amb}")
    action_amb = uci_to_action_int(uci_amb)
    print(f"  Encoded Action: {action_amb}")

    if action_amb is not None:
        decoded_amb = action_int_to_uci(action_amb)
        print(f"  Decoded UCI: {decoded_amb}")
        # Note: Decoding might infer 'q', but encoding shouldn't if 'q' wasn't specified

        # Simulate the move using the action integer (if encoded)
        next_board_amb = game.get_next_state(board.copy(), action_amb, 1)
        print("  Board after move:")
        print(next_board_amb)
        promoted_piece_amb = next_board_amb.piece_at(chess.H8)
        print(f"  Piece at h8: {promoted_piece_amb}") # Should likely be None or still Pawn
    else:
        print("  Encoding failed (as expected for ambiguous promo in strict encoding).")
    print("-" * 20)

if __name__ == "__main__":
    test_promotion_scenario() 