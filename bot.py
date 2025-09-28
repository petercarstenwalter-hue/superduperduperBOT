# bot.py
import os
API_TOKEN = os.getenv("LICHESS_TOKEN")
import math
import time
import traceback
import chess
import berserk

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
    return chess.square_rank(sq)

def square_file(sq):
    return chess.square_file(sq)

def king_ring_sq(kingsq):
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
# Evaluation
# ---------------------------
def evaluate_board(board: chess.Board) -> float:
    if board.is_checkmate():
        return math.inf if board.turn == BLACK else -math.inf
    if board.is_stalemate():
        return 0.0

    score = 0.0
    files_white = [0] * 8
    files_black = [0] * 8
    for sq in board.pieces(chess.PAWN, WHITE):
        files_white[square_file(sq)] += 1
    for sq in board.pieces(chess.PAWN, BLACK):
        files_black[square_file(sq)] += 1
    for f in range(8):
        if files_white[f] > 1:
            score -= (files_white[f] - 1)
        if files_black[f] > 1:
            score += (files_black[f] - 1)

    white_pawn_attacks = pawn_diagonal_attacks(board, WHITE)
    black_pawn_attacks = pawn_diagonal_attacks(board, BLACK)
    for sq in chess.SQUARES:
        wc = len(board.attackers(WHITE, sq))
        bc = len(board.attackers(BLACK, sq))
        if sq in black_pawn_attacks:
            wc = 0
        if sq in white_pawn_attacks:
            bc = 0
        score += (wc - bc)

    for sq in board.pieces(chess.PAWN, WHITE):
        r = square_rank(sq)
        score += 0.2 * (7 - r)
    for sq in board.pieces(chess.PAWN, BLACK):
        r = square_rank(sq)
        score -= 0.2 * (r)

    bk = board.king(BLACK)
    wk = board.king(WHITE)
    if bk is not None:
        for sq in king_ring_sq(bk):
            if board.attackers(WHITE, sq):
                score += 2
    if wk is not None:
        for sq in king_ring_sq(wk):
            if board.attackers(BLACK, sq):
                score -= 2

    for color in (WHITE, BLACK):
        pawns = list(board.pieces(chess.PAWN, color))
        files = [square_file(sq) for sq in pawns]
        for sq in pawns:
            f = square_file(sq)
            left, right = f - 1, f + 1
            if not ((left >= 0 and left in files) or (right <= 7 and right in files)):
                score -= 1 if color == WHITE else -1

    for sq in board.pieces(chess.PAWN, WHITE):
        r, f = square_rank(sq), square_file(sq)
        if r < 7:
            adj_support = any(
                (test_sq in board.pieces(chess.PAWN, WHITE))
                for df in (-1, 1)
                for r2 in range(r + 1, 8)
                if 0 <= f + df <= 7
                for test_sq in [chess.square(f + df, r2)]
            )
            front_sq = chess.square(f, r + 1)
            if not adj_support and board.attackers(BLACK, front_sq):
                score -= 1

    for sq in board.pieces(chess.PAWN, BLACK):
        r, f = square_rank(sq), square_file(sq)
        if r > 0:
            adj_support = any(
                (test_sq in board.pieces(chess.PAWN, BLACK))
                for df in (-1, 1)
                for r2 in range(r - 1, -1, -1)
                if 0 <= f + df <= 7
                for test_sq in [chess.square(f + df, r2)]
            )
            front_sq = chess.square(f, r - 1)
            if not adj_support and board.attackers(WHITE, front_sq):
                score += 1

    return float(score)

# ---------------------------
# Dynamic search depth
# ---------------------------
def compute_dynamic_depth(board: chess.Board, base_depth=BASE_SEARCH_DEPTH) -> int:
    depth = base_depth
    current_counts = {WHITE: {}, BLACK: {}}
    for color in (WHITE, BLACK):
        for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            current_counts[color][p] = len(board.pieces(p, color))
    for color in (WHITE, BLACK):
        for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            depth += (STARTING_COUNTS[color][p] - current_counts[color][p])
        depth += 2 * (STARTING_COUNTS[color][chess.QUEEN] - current_counts[color][chess.QUEEN])
    for color in (WHITE, BLACK):
        for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            extra = current_counts[color][p] - STARTING_COUNTS[color][p]
            if extra > 0:
                depth -= 2 * extra if p == chess.QUEEN else 1 * extra
    return max(1, int(depth))

# ---------------------------
# Negamax search
# ---------------------------
def negamax(board: chess.Board, depth: int, alpha: float, beta: float, pruning_threshold: int) -> float:
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    max_eval = -math.inf
    moves = sorted(board.legal_moves, key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)
    use_pruning = depth >= pruning_threshold

    for move in moves:
        board.push(move)
        val = -negamax(board, depth - 1, -beta, -alpha, pruning_threshold)
        board.pop()
        max_eval = max(max_eval, val)
        if use_pruning:
            alpha = max(alpha, val)
            if alpha >= beta:
                break
    return max_eval

def select_best_move(board: chess.Board):
    dynamic_depth = compute_dynamic_depth(board)
    extra_ply = max(0, dynamic_depth - BASE_SEARCH_DEPTH)
    pruning_threshold = 4 + (extra_ply // 5) * 2

    best_move, best_score = None, -math.inf
    for move in sorted(board.legal_moves, key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True):
        board.push(move)
        score = -negamax(board, dynamic_depth - 1, -math.inf, math.inf, pruning_threshold)
        board.pop()
        if score > best_score:
            best_score, best_move = score, move
    return best_move, best_score, dynamic_depth

# ---------------------------
# Lichess integration (fixed)
# ---------------------------
def run_lichess_bot():
    token = os.getenv("LICHESS_TOKEN")
    if not token:
        print("ERROR: LICHESS_TOKEN not set.")
        return

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)
    bot_id = client.account.get()["id"]
    print(f"🤖 Logged in as {bot_id}")

    for event in client.bots.stream_incoming_events():
        try:
            if event["type"] == "challenge":
                ch_id = event["challenge"]["id"]
                client.challenges.accept(ch_id)
                print(f"✅ Accepted challenge {ch_id}")

            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]
                print(f"🎮 Game started {game_id}")
                board = chess.Board()
                game_full = None
                for state in client.bots.stream_game_state(game_id):
                    if state["type"] == "gameFull":
                        game_full = state
                        moves = state["state"].get("moves", "")
                        if moves:
                            for mv in moves.split():
                                board.push_uci(mv)
                        white_id = state["white"]["id"]
                        black_id = state["black"]["id"]
                        our_color = WHITE if white_id == bot_id else BLACK
                        print(f"🟢 We are {'White' if our_color == WHITE else 'Black'}")

                    elif state["type"] == "gameState":
                        moves = state.get("moves", "")
                        board = chess.Board()
                        if moves:
                            for mv in moves.split():
                                board.push_uci(mv)

                        if board.is_game_over():
                            print(f"🏁 Game over ({board.result()})")
                            break

                        if board.turn == our_color:
                            move, score, depth = select_best_move(board)
                            if move:
                                client.bots.make_move(game_id, move.uci())
                                print(f"♟️ Played {move.uci()} (eval {score:.2f}, depth {depth})")
        except Exception as e:
            print("⚠️ Error handling event:", e)
            traceback.print_exc()

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    try:
        run_lichess_bot()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print("Unhandled exception:", exc)
        traceback.print_exc()
