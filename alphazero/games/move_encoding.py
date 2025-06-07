import math

# --- Constants and Mappings --- (Same as before)
SQUARE_MAP = {f'{chr(ord("a")+f)}{r+1}': r*8+f for r in range(8) for f in range(8)}
INV_SQUARE_MAP = {v: k for k, v in SQUARE_MAP.items()}
QUEEN_DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
QUEEN_DIR_MAP = {direction: i for i, direction in enumerate(QUEEN_DIRECTIONS)}
KNIGHT_OFFSETS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
KNIGHT_OFFSET_MAP = {offset: i for i, offset in enumerate(KNIGHT_OFFSETS)}
# UNDERPROMO_DIRECTIONS and UNDERPROMO_DIR_MAP are no longer used directly in encoding check
UNDERPROMO_PIECES = ['n', 'b', 'r']
UNDERPROMO_PIECE_MAP = {piece: i for i, piece in enumerate(UNDERPROMO_PIECES)}
INV_UNDERPROMO_PIECE_MAP = {v: k for k, v in UNDERPROMO_PIECE_MAP.items()}
NUM_SQUARES = 64
NUM_PLANES = 73
ACTION_SIZE = NUM_PLANES * NUM_SQUARES # 4672

# --- Helper Functions --- (Same as before)
def square_to_int(sq_name):
    return SQUARE_MAP.get(sq_name)
def int_to_square(sq_idx):
    return INV_SQUARE_MAP.get(sq_idx)
def get_coords(sq_idx):
    if sq_idx is None: return None, None
    return sq_idx // 8, sq_idx % 8
def get_sq_idx(rank, file):
    if 0 <= rank < 8 and 0 <= file < 8:
        return rank * 8 + file
    return None

# --- Encoding (Corrected Underpromotion) ---
def uci_to_action_int(uci_move):
    if not isinstance(uci_move, str) or len(uci_move) < 4 or len(uci_move) > 5: return None
    from_sq_name = uci_move[0:2]
    to_sq_name = uci_move[2:4]
    promotion_piece = uci_move[4] if len(uci_move) == 5 else None
    from_idx = square_to_int(from_sq_name)
    to_idx = square_to_int(to_sq_name)
    if from_idx is None or to_idx is None: return None

    dest_sq_idx = to_idx
    from_r, from_f = get_coords(from_idx)
    to_r, to_f = get_coords(to_idx)
    dr = to_r - from_r
    dc = to_f - from_f

    plane_idx = -1

    # 1. Handle Underpromotions
    if promotion_piece in UNDERPROMO_PIECES:
        dir_type_idx = -1
        # Determine the relative move type index (0:Left, 1:Fwd, 2:Right)
        if dr == -1: # White pawn move (N, NW, NE)
            if dc == -1: dir_type_idx = 0 # NW -> Left Type
            elif dc == 0: dir_type_idx = 1 # N  -> Fwd Type
            elif dc == 1: dir_type_idx = 2 # NE -> Right Type
        elif dr == 1: # Black pawn move (S, SW, SE)
            if dc == -1: dir_type_idx = 2 # SW -> Right Type (relative to Black)
            elif dc == 0: dir_type_idx = 1 # S  -> Fwd Type
            elif dc == 1: dir_type_idx = 0 # SE -> Left Type (relative to Black)

        if dir_type_idx != -1:
            piece_index = UNDERPROMO_PIECE_MAP[promotion_piece]
            plane_idx = 64 + dir_type_idx * 3 + piece_index
        else:
            # Should not happen if it's a valid pawn move promotion
            print(f"Warning: Invalid pawn direction ({dr},{dc}) for underpromotion {uci_move}")
            return None

    # 2. Check for Knight Moves (only if not an underpromotion)
    elif promotion_piece is None:
        rel_offset = (-dr, -dc) # Knight offsets defined from dest to origin
        if rel_offset in KNIGHT_OFFSET_MAP:
            knight_index = KNIGHT_OFFSET_MAP[rel_offset]
            plane_idx = 56 + knight_index

    # 3. Check for Queen-like Moves (if not knight or underpromo)
    # Includes regular moves and Queen promotions
    if plane_idx == -1:
        # Check if straight or diagonal
        if dr == 0 or dc == 0 or abs(dr) == abs(dc):
            distance = max(abs(dr), abs(dc))
            if distance == 0: return None # Cannot move 0 squares

            # Queen moves plane only covers distances 1-7
            if distance > 7:
                # Check if it's a valid promotion not covered by underpromo (i.e., Queen)
                is_queen_promo = (promotion_piece == 'q')
                is_pawn_move = (abs(dr) == 1 and abs(dc) <= 1) or \
                               (abs(dr) == 2 and dc == 0 and (from_r == 1 or from_r == 6)) # Include double push

                if not (is_queen_promo and is_pawn_move and distance == 1): # Only allow dist > 7 if not queen promo pawn move
                     # It's not a standard queen/knight/underpromo move pattern encoded here
                     # print(f"Warning: Move distance {distance} > 7 for non-promo or invalid move {uci_move}")
                     # Allow pawn double push (dist 2) to be handled below
                     if not (distance == 2 and dc == 0 and abs(dr)==2):
                         print(f"Warning: Move {uci_move} has distance {distance} > 7 or invalid geometry.")
                         return None # Treat as invalid encoding if dist > 7 unless pawn jump

            # Clamp distance for encoding purposes if it was pawn jump > 7 logic isn't perfect
            encoded_distance = min(distance, 7)
            if encoded_distance == 0: return None # Should be caught above

            # Normalized direction
            norm_dr = dr // distance # Use original distance for normalization
            norm_dc = dc // distance
            move_dir = (norm_dr, norm_dc)

            if move_dir in QUEEN_DIR_MAP:
                dir_index = QUEEN_DIR_MAP[move_dir]
                plane_idx = dir_index * 7 + (encoded_distance - 1)
            else:
                print(f"Error: Unrecognized queen-like direction {move_dir} for {uci_move}")
                return None
        elif plane_idx == -1 and promotion_piece is None:
             # If it wasn't underpromo, knight, or queen-like, it's invalid for this scheme
             # e.g. trying to encode something like a King moving 3 squares
             print(f"Warning: Move {uci_move} doesn't fit known categories (dr={dr}, dc={dc})")
             return None


    # Combine plane and destination square
    if 0 <= plane_idx < NUM_PLANES and 0 <= dest_sq_idx < NUM_SQUARES:
        action_int = plane_idx * NUM_SQUARES + dest_sq_idx
        # --- Add safety check ---
        if action_int >= ACTION_SIZE:
            print(f"CRITICAL ERROR: Calculated action_int {action_int} is out of bounds for {uci_move} (plane={plane_idx}, dest_sq={dest_sq_idx}). Returning None.")
            return None
        # --- End safety check ---
        return action_int
    else:
        # Enhanced error message
        print(f"Error: Failed to determine valid plane ({plane_idx}) or destination ({dest_sq_idx}) for move {uci_move}")
        return None


# --- Decoding (Corrected Underpromotion) ---
def action_int_to_uci(action_int):
    """
    Converts an action integer (0-4671) back to a potential UCI move string.
    Corrected underpromotion decoding logic.
    """
    if not isinstance(action_int, int) or not (0 <= action_int < ACTION_SIZE): return None

    plane_idx = action_int // NUM_SQUARES
    dest_sq_idx = action_int % NUM_SQUARES

    to_sq_name = int_to_square(dest_sq_idx)
    if to_sq_name is None: return None
    to_r, to_f = get_coords(dest_sq_idx)
    promotion_char = ""
    from_r, from_f = -1, -1

    # 1. Decode Underpromotions (Planes 64-72)
    if 64 <= plane_idx <= 72:
        underpromo_sub_idx = plane_idx - 64
        dir_type_idx = underpromo_sub_idx // 3 # 0:Left, 1:Fwd, 2:Right (relative types)
        piece_index = underpromo_sub_idx % 3
        promotion_char = INV_UNDERPROMO_PIECE_MAP[piece_index]
        move_dir_dr, move_dir_dc = 0, 0

        # Infer actual move direction based on destination rank
        # *** CORRECTED RANK CHECK LOGIC ***
        if to_r == 7: # BLACK promotion (arrived on rank 8) - needs move like (1, dc) (S, SE, SW)
            if dir_type_idx == 0: move_dir_dr, move_dir_dc = 1, 1   # Type 0 (Left rel. to Black) -> SE
            elif dir_type_idx == 1: move_dir_dr, move_dir_dc = 1, 0   # Type 1 (Fwd rel. to Black) -> S
            elif dir_type_idx == 2: move_dir_dr, move_dir_dc = 1, -1  # Type 2 (Right rel. to Black) -> SW
            else: return None # Should not happen
        elif to_r == 0: # WHITE promotion (arrived on rank 1) - needs move like (-1, dc) (N, NW, NE)
            if dir_type_idx == 0: move_dir_dr, move_dir_dc = -1, -1 # Type 0 (Left rel. to White) -> NW
            elif dir_type_idx == 1: move_dir_dr, move_dir_dc = -1, 0  # Type 1 (Fwd rel. to White) -> N
            elif dir_type_idx == 2: move_dir_dr, move_dir_dc = -1, 1  # Type 2 (Right rel. to White) -> NE
            else: return None # Should not happen
        else:
            # Promotion action encoded for a non-promotion square? Invalid.
             return None

        from_r = to_r - move_dir_dr
        from_f = to_f - move_dir_dc

    # 2. Decode Knight Moves (Planes 56-63)
    elif 56 <= plane_idx <= 63:
        knight_index = plane_idx - 56
        offset_dr, offset_dc = KNIGHT_OFFSETS[knight_index] # dest -> origin offset
        from_r = to_r + offset_dr
        from_f = to_f + offset_dc

    # 3. Decode Queen-like Moves (Planes 0-55)
    elif 0 <= plane_idx <= 55:
        dir_index = plane_idx // 7
        distance = (plane_idx % 7) + 1
        move_dir_dr, move_dir_dc = QUEEN_DIRECTIONS[dir_index] # Actual move direction
        from_r = to_r - (move_dir_dr * distance)
        from_f = to_f - (move_dir_dc * distance)

        # --- REMOVED HEURISTIC QUEEN PROMOTION CHECK ---
        # The promotion_char should only be set for underpromotion planes (64-72)
        # The game logic (get_next_state) will handle default Queen promotion based on state.
        # promotion_char remains ""

    else: return None # Invalid plane

    # Final check: Ensure calculated origin is on the board
    from_sq_idx = get_sq_idx(from_r, from_f)
    if from_sq_idx is None:
        # Action integer maps to a move starting off-board.
        return None

    from_sq_name = int_to_square(from_sq_idx)
    return f"{from_sq_name}{to_sq_name}{promotion_char}"


if __name__ == "__main__":
    # --- Example Usage (Same test cases) ---
    uci_moves = ["e2e4", "g1f3", "a2a4", "b8c6", "h7h8q", "a7a8r", "e1g1", "e8c8", "f3d4", "c6d4"]
    encoded_actions = []

    print("--- Encoding ---")
    for move in uci_moves:
        action = uci_to_action_int(move)
        encoded_actions.append(action)
        print(f"{move:>6} -> {action}")

    print("\n--- Decoding ---")
    for i, action in enumerate(encoded_actions):
        if action is not None:
            # Use the corrected decoder function name
            decoded_move = action_int_to_uci(action)
            original_move = uci_moves[i]
            status = "OK" if decoded_move == original_move else f"MISMATCH (Got: {decoded_move})"
            print(f"{action:>4} -> {decoded_move:<6} ({status})")
        else:
            print(f"Skipped decoding for original move: {uci_moves[i]} which encoded to None")

    print("\n--- Edge Cases ---")
    edge_moves = ['a7a8n', 'h2h1r', 'e7e5', 'b2b1b'] # Added white underpromo example
    for move in edge_moves:
        action = uci_to_action_int(move)
        print(f"{move:>6} -> {action}")
        if action is not None:
            decoded_move = action_int_to_uci(action)
            status = "OK" if decoded_move == move else f"MISMATCH (Got: {decoded_move})"
            print(f"{action:>4} -> {decoded_move:<6} ({status})")

    # Example: Decode a specific action int that was problematic before
    print(f"\nDecoding action 4472 (expected a7a8r): {action_int_to_uci(4472)}")
    print(f"Decoding action 4423 (expected h2h1r): {action_int_to_uci(4423)}")
    print(f"Decoding action 2869 (expected None): {action_int_to_uci(2869)}")