class State:
    def __init__(self):
        self.board = [["" for _ in range(3)] for _ in range(3)]

    def make_move(self, i, j, sign):
        self.board[i][j] = sign

    def get_sign(self, i, j):
        return self.board[i][j]

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != "":
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != "":
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != "":
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != "":
            return True
        return False

    def check_draw(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == "":
                    return False
        return True

    def possible_moves(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == "":
                    yield i, j

    def possible_states(self, player):
        for move in self.possible_moves():
            new_state = self.copy()
            new_state.make_move(move[0], move[1], player)
            yield new_state

    def copy(self):
        new_state = State()
        new_state.board = [row.copy() for row in self.board]
        return new_state

    def print_board(self):
        for i in range(3):
            for j in range(3):
                sign = self.get_sign(i, j) if self.get_sign(i, j) != "" else " "
                print(sign, end=" ")
            print()
