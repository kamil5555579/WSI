def minimax(state, depth, max_move):

    if state.check_winner() or depth == 0:
        return heuristic(state)
    if state.check_draw():
        return 0

    possible_states = state.possible_states("O" if max_move else "X")

    if max_move:
        best_score = float("-inf")
        for state_copy in possible_states:
            best_score = max(best_score, minimax(state_copy, depth - 1, False))
        return best_score
    else:
        best_score = float("inf")
        for state_copy in possible_states:
            best_score = min(best_score, minimax(state_copy, depth - 1, True))
        return best_score


def minimax_alpha_beta(state, depth, alpha, beta, max_move):

    if state.check_winner() or depth == 0:
        return heuristic(state)
    if state.check_draw():
        return 0

    possible_states = state.possible_states("O" if max_move else "X")

    if max_move:
        best_score = float("-inf")
        for state_copy in possible_states:
            best_score = max(
                best_score,
                minimax_alpha_beta(state_copy, depth - 1, alpha, beta, False),
            )
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = float("inf")
        for state_copy in possible_states:
            best_score = min(
                best_score, minimax_alpha_beta(state_copy, depth - 1, alpha, beta, True)
            )
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score


# player O is maximizing player
def heuristic(state):
    if state.check_winner():
        for i in range(3):
            if (
                state.get_sign(i, 0)
                == state.get_sign(i, 1)
                == state.get_sign(i, 2)
                == "X"
            ):
                return -10
            if (
                state.get_sign(i, 0)
                == state.get_sign(i, 1)
                == state.get_sign(i, 2)
                == "O"
            ):
                return 10
            if (
                state.get_sign(0, i)
                == state.get_sign(1, i)
                == state.get_sign(2, i)
                == "X"
            ):
                return -10
            if (
                state.get_sign(0, i)
                == state.get_sign(1, i)
                == state.get_sign(2, i)
                == "O"
            ):
                return 10

        if (
            state.get_sign(0, 0) == state.get_sign(1, 1) == state.get_sign(2, 2) == "X"
            or state.get_sign(0, 2)
            == state.get_sign(1, 1)
            == state.get_sign(2, 0)
            == "X"
        ):
            return -10
        if (
            state.get_sign(0, 0) == state.get_sign(1, 1) == state.get_sign(2, 2) == "O"
            or state.get_sign(0, 2)
            == state.get_sign(1, 1)
            == state.get_sign(2, 0)
            == "O"
        ):
            return 10

    else:
        heuristic = 0
        positions_values = {
            (0, 0): 3,
            (0, 1): 2,
            (0, 2): 3,
            (1, 0): 2,
            (1, 1): 4,
            (1, 2): 2,
            (2, 0): 3,
            (2, 1): 2,
            (2, 2): 3,
        }
        for i in range(3):
            for j in range(3):
                if state.get_sign(i, j) == "X":
                    heuristic -= positions_values[(i, j)]
                elif state.get_sign(i, j) == "O":
                    heuristic += positions_values[(i, j)]
    return heuristic
