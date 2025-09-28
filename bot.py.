# bot.py
import os
API_TOKEN = os.getenv("LICHESS_TOKEN")
import math
import time
import traceback
import chess
import berserk
import requests

# ---------------------------
# Configuration / Constants
# ---------------------------
BASE_SEARCH_DEPTH = 10  # ply at start of game
WHITE = chess.WHITE
BLACK = chess.BLACK

STARTING_COUNTS = {
    WHITE: {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1},
    BLACK: {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
}

# ---------------------------
# Helpers
# ---------------------------
def square_rank(sq):
    return chess.square_rank(sq)  # 0..7

def square_file(sq):
    return chess.square_file(sq)  # 0..7

def king_ring_sq(kingsq):
    """Return list of squares adjacent to kingsq (8-neighborhood)."""
    ring = []
    kr = square_rank(kingsq)
    kf = square_file(kingsq)
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr = kr + dr
            nf = kf + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                ring.append(chess.square(nf, nr))
    return ring

def pawn_diagonal_attacks(board, color):
    """Return a set of squares diagonally attacked by pawns of given color."""
    attacks = set()
    for sq in board.pieces(chess.PAWN, color):
        r = square_rank(sq)
        f = square_file(sq)
        if color == WHITE:
            nr = r + 1
            for nf in (f - 1, f + 1):
                if 0 <= nr <= 7 and 0 <= nf <= 7:
                    attacks.add(chess.square(nf, nr))
        else:
            nr = r - 1
            for nf in (f - 1, f + 1):
                if 0 <= nr <= 7 and 0 <= nf <= 7:
                    attacks.add(chess.square(nf, nr))
    return attacks

# ---------------------------
# Evaluation (implements your rules)
# ---------------------------
def evaluate_board(board: chess.Board) -> float:
    """
    Return a score where + is good for White, - is good for Black.
    Implements:
      - doubled pawns: white -1 per extra pawn on file, black +1 per extra pawn on file
      - square control: +1 per square a white piece controls, -1 per square a black piece controls
        * BUT neutralize white control if that square is diagonally controlled by a black pawn,
          and neutralize black control if that square is diagonally controlled by a white pawn.
      - pawn progress: white +(7 - rank)*0.2, black -(rank)*0.2
      - squares around kings controlled by opponent: +2 for white control near black king,
        -2 for black control near white king
      - isolated pawns: white -1 each, black +1 each
      - backwards pawns: white -1 each, black +1 each (heuristic)
      - stalemate -> 0; checkmate -> +/- inf
    """

    # Terminal conditions
    if board.is_checkmate():
        # side to move is checkmated
        if board.turn == BLACK:
            return math.inf  # black king checkmated -> +inf (good for white)
        else:
            return -math.inf
    if board.is_stalemate():
        return 0.0

    score = 0.0

    # --------------------
    # 1) Doubled pawns
    # --------------------
    files_white = [0] * 8
    files_black = [0] * 8
    for sq in board.pieces(chess.PAWN, WHITE):
        files_white[square_file(sq)] += 1
    for sq in board.pieces(chess.PAWN, BLACK):
        files_black[square_file(sq)] += 1
    for f in range(8):
        if files_white[f] > 1:
            score -= 1 * (files_white[f] - 1)
        if files_black[f] > 1:
            score += 1 * (files_black[f] - 1)

    # --------------------
    # 2) Control of squares (with pawn-diagonal neutralization)
    # --------------------
    white_pawn_attacks = pawn_diagonal_attacks(board, WHITE)
    black_pawn_attacks = pawn_diagonal_attacks(board, BLACK)

    square_control_score = 0.0
    for sq in chess.SQUARES:
        white_ctrl = len(board.attackers(WHITE, sq))
        black_ctrl = len(board.attackers(BLACK, sq))

        wc = white_ctrl
        bc = black_ctrl

        # Neutralization per your spec:
        # if a white piece controls a square that a black pawn diagonally controls -> white contribution = 0
        if sq in black_pawn_attacks:
            wc = 0
        # if a black piece controls a square that a white pawn diagonally controls -> black contribution = 0
        if sq in white_pawn_attacks:
            bc = 0

        square_control_score += (wc - bc)

    score += square_control_score

    # --------------------
    # 3) Pawn proximity to promotion
    # --------------------
    pawn_progress_score = 0.0
    for sq in board.pieces(chess.PAWN, WHITE):
        r = square_rank(sq)
        pawn_progress_score += 0.2 * (7 - r)
    for sq in board.pieces(chess.PAWN, BLACK):
        r = square_rank(sq)
        pawn_progress_score -= 0.2 * (r)
    score += pawn_progress_score

    # --------------------
    # 4) Squares around kings controlled by opponent
    # --------------------
    bk = board.king(BLACK)
    wk = board.king(WHITE)
    if bk is not None:
        for sq in king_ring_sq(bk):
            if len(board.attackers(WHITE, sq)) > 0:
                score += 2
    if wk is not None:
        for sq in king_ring_sq(wk):
            if len(board.attackers(BLACK, sq)) > 0:
                score -= 2

    # --------------------
    # 5) Isolated pawns
    # --------------------
    for color in (WHITE, BLACK):
        pawns = list(board.pieces(chess.PAWN, color))
        files = [square_file(sq) for sq in pawns]
        for sq in pawns:
            f = square_file(sq)
            left = f - 1
            right = f + 1
            has_adjacent = False
            if left >= 0 and left in files:
                has_adjacent = True
            if right <= 7 and right in files:
                has_adjacent = True
            if not has_adjacent:
                if color == WHITE:
                    score -= 1
                else:
                    score += 1

    # --------------------
    # 6) Backward pawns (heuristic)
    #    - a pawn is backward if it has no friendly pawn on adjacent files ahead
    #      and its forward square is attacked by the opponent (so it cannot safely advance)
    # --------------------
    # white backwards
    for sq in board.pieces(chess.PAWN, WHITE):
        r = square_rank(sq)
        f = square_file(sq)
        if r == 7:
            continue
        adj_support = False
        for df in (-1, 1):
            nf = f + df
            if 0 <= nf <= 7:
                for r2 in range(r + 1, 8):
                    test_sq = chess.square(nf, r2)
                    if test_sq in board.pieces(chess.PAWN, WHITE):
                        adj_support = True
                        break
                if adj_support:
                    break
        front_sq = chess.square(f, r + 1)
        front_attacked_by_black = len(board.attackers(BLACK, front_sq)) > 0
        if (not adj_support) and front_attacked_by_black:
            score -= 1

    # black backwards
    for sq in board.pieces(chess.PAWN, BLACK):
        r = square_rank(sq)
        f = square_file(sq)
        if r == 0:
            continue
        adj_support = False
        for df in (-1, 1):
            nf = f + df
            if 0 <= nf <= 7:
                for r2 in range(r - 1, -1, -1):
                    test_sq = chess.square(nf, r2)
                    if test_sq in board.pieces(chess.PAWN, BLACK):
                        adj_support = True
                        break
                if adj_support:
                    break
        front_sq = chess.square(f, r - 1)
        front_attacked_by_white = len(board.attackers(WHITE, front_sq)) > 0
        if (not adj_support) and front_attacked_by_white:
            score += 1

    return float(score)

# ---------------------------
# Dynamic search depth calculation (per your rules)
# ---------------------------
def compute_dynamic_depth(board: chess.Board, base_depth=BASE_SEARCH_DEPTH) -> int:
    """
    - base 10 ply at start
    - for every knight/bishop/rook that comes off the board (for both sides) +1 ply
    - for every queen off the board +2 ply
    - after pawn promotes to a queen: subtract 2 ply per such promotion
    - after pawn promotes to any other piece: subtract 1 ply per such promotion
    Promotions detected by comparing current piece counts to STARTING_COUNTS.
    """
    depth = base_depth

    # Current counts
    current_counts = {WHITE: {}, BLACK: {}}
    for color in (WHITE, BLACK):
        for ptype in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            current_counts[color][ptype] = len(board.pieces(ptype, color))

    # For knights, bishops, rooks: missing from starting -> +1 per missing
    for color in (WHITE, BLACK):
        for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            missing = STARTING_COUNTS[color][p] - current_counts[color][p]
            if missing > 0:
                depth += missing * 1

    # Queens missing -> +2 per missing
    for color in (WHITE, BLACK):
        missing_q = STARTING_COUNTS[color][chess.QUEEN] - current_counts[color][chess.QUEEN]
        if missing_q > 0:
            depth += missing_q * 2

    # Promotions: additional pieces beyond starting counts -> subtract ply
    # Count extras beyond starting counts for queen/rook/bishop/knight
    for color in (WHITE, BLACK):
        for promoted_ptype in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            extra = current_counts[color][promoted_ptype] - STARTING_COUNTS[color][promoted_ptype]
            if extra > 0:
                if promoted_ptype == chess.QUEEN:
                    depth -= 2 * extra
                else:
                    depth -= 1 * extra

    # Enforce minimum depth of 1 ply
    if depth < 1:
        depth = 1
    return int(depth)

# ---------------------------
# Search (negamax with conditional alpha-beta pruning)
# ---------------------------
def negamax(board: chess.Board, depth: int, alpha: float, beta: float, color_multiplier: int, pruning_threshold: int) -> float:
    """
    Negamax style search:
      - color_multiplier: 1 when evaluating from White perspective, -1 otherwise
      - only perform alpha-beta pruning if depth >= pruning_threshold
    Returns evaluation from White's perspective.
    """
    # Terminal or depth 0 evaluation
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    max_eval = -math.inf
    legal_moves = list(board.legal_moves)

    # Basic move ordering: captures and checks first
    def move_key(m):
        return (board.is_capture(m), board.gives_check(m))
    legal_moves.sort(key=move_key, reverse=True)

    use_pruning = (depth >= pruning_threshold)
    for move in legal_moves:
        board.push(move)
        val = -negamax(board, depth - 1, -beta, -alpha, -color_multiplier, pruning_threshold)
        board.pop()

        if val > max_eval:
            max_eval = val

        if use_pruning:
            if val > alpha:
                alpha = val
            if alpha >= beta:
                # prune
                return max_eval

    # No legal moves -> should be handled earlier; return evaluation if occurs
    if max_eval == -math.inf:
        return evaluate_board(board)
    return max_eval

def select_best_move(board: chess.Board):
    """
    Compute dynamic depth and pruning threshold, then evaluate each legal move
    and choose the best move (negamax).
    Also logs evaluations for each considered move.
    """
    dynamic_depth = compute_dynamic_depth(board)
    extra_ply = max(0, dynamic_depth - BASE_SEARCH_DEPTH)
    extra_prune_steps = (extra_ply // 5) * 2
    pruning_threshold = 4 + extra_prune_steps

    legal_moves = list(board.legal_moves)
    # Move ordering
    legal_moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)

    best_move = None
    best_score = -math.inf if board.turn == WHITE else math.inf

    print(f"\n🧠 Thinking at depth {dynamic_depth} ply (prune threshold {pruning_threshold})...")
    for move in legal_moves:
        board.push(move)
        score = -negamax(board, dynamic_depth - 1, -math.inf, math.inf, -1, pruning_threshold)
        board.pop()
        # Log the move evaluation
        print(f"   Move {move.uci()} -> Eval {score:.3f}")
        if board.turn == WHITE:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move

    # If nothing found (shouldn't happen), pick a legal move
    if best_move is None and legal_moves:
        best_move = legal_moves[0]

    print(f"✅ Chosen move: {best_move.uci()}  (eval {best_score:.3f})")
    return best_move, best_score, dynamic_depth, pruning_threshold

# ---------------------------
# Lichess integration
# ---------------------------
def run_lichess_bot():
    token = os.getenv("LICHESS_TOKEN")
    if not token:
        print("ERROR: LICHESS_TOKEN environment variable not set.")
        return

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    print("🔌 Lichess bot starting up... (auto-accepting all challenges)")

    # Helper to accept challenge
    def accept_challenge(ch_id):
        try:
            client.challenges.accept(ch_id)
            print(f"✅ Accepted challenge {ch_id}")
        except Exception as e:
            print("Error accepting challenge:", e)

    # Stream incoming events (challenges, gameStart)
    try:
        for event in client.bots.stream_incoming_events():
            try:
                typ = event.get("type")
                if typ == "challenge":
                    # Auto-accept any incoming challenge
                    ch_id = event["challenge"]["id"]
                    print("Incoming challenge:", ch_id, "from", event["challenge"]["challenger"]["id"])
                    accept_challenge(ch_id)

                elif typ == "gameStart":
                    game_id = event["game"]["id"]
                    print("🎮 Game started:", game_id)
                    # stream game state and play
                    board = chess.Board()
                    # stream_game_state yields events: gameFull and subsequent gameState events
                    for state in client.bots.stream_game_state(game_id):
                        try:
                            stype = state.get("type")
                            # If the stream reports game over, break
                            if state.get("status") in ("mate", "resign", "draw", "stalemate", "timeout"):
                                print("🏁 Game over:", state.get("status"))
                                break

                            if stype == "gameFull":
                                moves_txt = state["state"].get("moves", "")
                                if moves_txt:
                                    for mv in moves_txt.split():
                                        board.push_uci(mv)
                            elif stype == "gameState":
                                moves_txt = state.get("moves", "")
                                # Update local board from move list
                                board = chess.Board()
                                if moves_txt:
                                    for mv in moves_txt.split():
                                        board.push_uci(mv)

                                # Is it our turn? Compare ids
                                try:
                                    account_info = client.account.get()
                                    our_id = account_info.get("id")
                                except Exception:
                                    # fallback: the gameStart event contains player ids in event["game"]
                                    our_id = None

                                white_id = state.get("white", {}).get("id") if state.get("white") else None
                                black_id = state.get("black", {}).get("id") if state.get("black") else None

                                its_our_turn = False
                                if board.turn == chess.WHITE and white_id == our_id:
                                    its_our_turn = True
                                if board.turn == chess.BLACK and black_id == our_id:
                                    its_our_turn = True

                                # If account lookup failed, try comparing against game info from earlier event data
                                # (some stream variants embed identity in event["game"])
                                # We'll also play when it's our turn by checking who the player is in the original gameStart event
                                if its_our_turn:
                                    # Select and send best move
                                    try:
                                        move, score, depth, prune = select_best_move(board)
                                        if move is None:
                                            # No move (should be game over), skip
                                            print("No move found; maybe game over.")
                                            continue
                                        client.bots.make_move(game_id, move.uci())
                                        print(f"♟️ Played {move.uci()} (eval {score:.3f}, depth {depth})")
                                    except Exception as e:
                                        print("Error selecting/making move:", e)
                                        traceback.print_exc()
                        except Exception as e:
                            print("Error while streaming game state:", e)
                            traceback.print_exc()
                    print("Finished game loop for", game_id)
                else:
                    # other event types
                    # print("Event:", typ)
                    pass
            except Exception as e:
                print("Error handling incoming event:", e)
                traceback.print_exc()
    except Exception as e:
        print("Fatal error in stream_incoming_events:", e)
        traceback.print_exc()

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    # Run the bot. If it crashes, print stack and exit (GitHub Actions can restart by schedule)
    try:
        run_lichess_bot()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print("Unhandled exception in bot:", exc)
        traceback.print_exc()
