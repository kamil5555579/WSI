import tkinter as tk
from tkinter import messagebox
import time

class TicTacToe:
    def __init__(self, master, minimax_depth=3):
        self.master = master
        self.minimax_depth = minimax_depth
        self.master.title("Tic Tac Toe")
        self.current_player = "X"
        self.board = [['' for _ in range(3)] for _ in range(3)]

        self.buttons = [[None] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(master, text="", font=("Arial", 20), width=5, height=2,
                                                command=lambda row=i, col=j: self.on_click(row, col))
                self.buttons[i][j].grid(row=i, column=j, padx=5, pady=5)
        
        self.computer_game_button = tk.Button(master, text="Computer move", font=("Arial", 10), width=15, height=2,
                                                command=self.computer_game)
        self.computer_game_button.grid(row=3, column=1, padx=5, pady=5)

    def on_click(self, row, col):
        if self.board[row][col] == "" and not self.check_winner(self.board):
            self.board[row][col] = self.current_player
            self.buttons[row][col].config(text=self.current_player)
            if self.check_winner(self.board):
                messagebox.showinfo("Winner", f"Player {self.current_player} wins!")
                self.reset_game()
            elif all(cell != "" for row in self.board for cell in row):
                messagebox.showinfo("Draw", "It's a draw!")
                self.reset_game()
            else:
                self.current_player = "O" if self.current_player == "X" else "X"


    def computer_move(self):
        # deep copy of board state
        current_board = [row.copy() for row in self.board]
        possible_moves = []
        for i in range(3):
            for j in range(3):
                if current_board[i][j] == "":
                    possible_moves.append((i, j))
        best_move = None
        best_score = float('-inf') if self.current_player == 'O' else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        for move in possible_moves:
            new_board = [row.copy() for row in current_board]
            new_board[move[0]][move[1]] = self.current_player
            if self.current_player == 'O':
                # score = self.minimax(new_board, self.minimax_depth, False)
                score = self.minimax_alpha_beta(new_board, self.minimax_depth, alpha, beta, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                # score = self.minimax(new_board, self.minimax_depth, True)
                score = self.minimax_alpha_beta(new_board, self.minimax_depth, alpha, beta, True)
                if score < best_score:
                    best_score = score
                    best_move = move
        print(best_move)
        print(best_score)
        self.board[best_move[0]][best_move[1]] = self.current_player
        self.buttons[best_move[0]][best_move[1]].config(text=self.current_player)
        if self.check_winner(self.board) and self.current_player == "O":
            messagebox.showinfo("Winner", f"Player O wins!")
            self.reset_game()
        elif self.check_winner(self.board) and self.current_player == "X":
            messagebox.showinfo("Winner", f"Player X wins!")
            self.reset_game()
        elif all(cell != "" for row in self.board for cell in row):
            messagebox.showinfo("Draw", "It's a draw!")
            self.reset_game()
        else:
            self.current_player = "X" if self.current_player == "O" else "O"

    def minimax(self, board, depth, max_move):
        if self.check_winner(board) or depth == 0:
            return self.heuristic(board)
        if all(cell != "" for row in board for cell in row):
            return 0
        successors = self.successors(board, 'O' if max_move else 'X')
        if max_move:
            best_score = float('-inf')
            for successor in successors:
                best_score = max(best_score, self.minimax(successor, depth - 1, False))
            return best_score
        else:
            best_score = float('inf')
            for successor in successors:
                best_score = min(best_score, self.minimax(successor, depth - 1, True))
            return best_score
        
    def minimax_alpha_beta(self, board, depth, alpha, beta, max_move):
        if self.check_winner(board) or depth == 0:
            return self.heuristic(board)
        if all(cell != "" for row in board for cell in row):
            return 0
        successors = self.successors(board, 'O' if max_move else 'X')
        if max_move:
            best_score = float('-inf')
            for successor in successors:
                best_score = max(best_score, self.minimax_alpha_beta(successor, depth - 1, alpha, beta, False))
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for successor in successors:
                best_score = min(best_score, self.minimax_alpha_beta(successor, depth - 1, alpha, beta, True))
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

    # player O is maximizing player
    def heuristic(self, board):
        if self.check_winner(board):
            for i in range(3):
                if board[i][0] == board[i][1] == board[i][2] == 'X':
                    return -10
                if board[i][0] == board[i][1] == board[i][2] == 'O':
                    return 10
                if board[0][i] == board[1][i] == board[2][i] == 'X':
                    return -10
                if board[0][i] == board[1][i] == board[2][i] == 'O':
                    return 10

            if board[0][0] == board[1][1] == board[2][2] == 'X' or \
                board[0][2] == board[1][1] == board[2][0] == 'X':
                return -10
            if board[0][0] == board[1][1] == board[2][2] == 'O' or \
                board[0][2] == board[1][1] == board[2][0] == 'O':
                return 10
        
        else:
            heuristic = 0
            positions_values = {
                (0, 0): 3, (0, 1): 2, (0, 2): 3,
                (1, 0): 2, (1, 1): 4, (1, 2): 2,
                (2, 0): 3, (2, 1): 2, (2, 2): 3
            }
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 'X':
                        heuristic -= positions_values[(i, j)]
                    elif board[i][j] == 'O':
                        heuristic += positions_values[(i, j)]
        return heuristic
        
    def successors(self, board, player):
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    new_board = [row.copy() for row in board]
                    new_board[i][j] = player
                    yield new_board

    def check_winner(self, board):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != "":
                return True
            if board[0][i] == board[1][i] == board[2][i] != "":
                return True
        if board[0][0] == board[1][1] == board[2][2] != "":
            return True
        if board[0][2] == board[1][1] == board[2][0] != "":
            return True
        return False

    def reset_game(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text="")
                self.board[i][j] = ""
        self.current_player = "X"

    def computer_game(self):
        self.computer_move()
        self.master.update()

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root, minimax_depth=3)
    root.mainloop()

