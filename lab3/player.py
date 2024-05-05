# tictactoe player
from minimax import minimax, minimax_alpha_beta


class Player:
    def __init__(self, player_type, player_sign):
        self.player_type = player_type
        self.player_sign = player_sign

    def get_move(self, state):
        if self.player_type == "human":
            return self.human_move(state)
        elif self.player_type == "computer":
            return self.computer_move(state)

    def human_move(self, state):
        chosen_field = input("Enter the field you want to place your sign in (1-9): ")
        move = (int(chosen_field) - 1) // 3, (int(chosen_field) - 1) % 3
        if move not in state.possible_moves():
            print("Field already taken")
            return self.human_move(state)
        return move

    def computer_move(self, state):
        best_move = None
        best_score = -float("inf") if self.player_sign == "O" else float("inf")
        possible_moves = state.possible_moves()
        for move in possible_moves:
            state_copy = state.copy()
            state_copy.make_move(move[0], move[1], self.player_sign)
            if self.player_sign == "O":
                # score = minimax(state_copy, 1, False)
                score = minimax_alpha_beta(
                    state_copy, 8, -float("inf"), float("inf"), False
                )
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                # score = minimax(state_copy, 1, True)
                score = minimax_alpha_beta(
                    state_copy, 8, -float("inf"), float("inf"), True
                )
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move
