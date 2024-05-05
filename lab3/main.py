from player import Player
from game import Game

player1 = Player("computer", "X")
player2 = Player("human", "O")
game = Game(player1, player2)
game.play()
