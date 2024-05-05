from state import State


class Game:

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.state = State()

    def play(self):
        player = self.player1
        while True:
            move = player.get_move(self.state)
            self.state.make_move(move[0], move[1], player.player_sign)
            self.state.print_board()
            if self.state.check_winner():
                print("Winner", player.player_sign)
                break
            if self.state.check_draw():
                print("Draw")
                break
            player = self.player1 if player == self.player2 else self.player2
