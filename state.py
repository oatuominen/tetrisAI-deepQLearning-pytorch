

import numpy as np
import constants as c
import copy
from piece import Piece

class State:

    def __init__(self):
        self.current_piece = Piece(c.O, (0, 0, 0))
        self.board = [[0] * c.COLUMNS for _ in range(c.ROWS)]
        self.color_board = {(x, y): (0, 0, 0) for x in range(20) for y in range(10)}
        self.score = 0
        self.level = 1
        self.rows_cleared = 0
        self.reward = 0.0
        self.current_reward = 0.0
        self.latest_position = []
        self.active_piece = False


    def to_id(self, board=None): 
        board = self.board if board is None else board
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        return [self.rows_cleared, holes, bumpiness, height]
    
    
    def stat_board(self, board=None):
        board = self.board if board is None else board
        board = [row[:] for row in self.board]
        for (row, col) in self.current_piece.positions:
            board[row][col] = 0
        return board


    def get_holes(self, board=None):
        board = self.board if board == None else board
        stationary_board = [row[:] for row in board]
        for (row, col) in self.current_piece.positions:
            stationary_board[row][col] = 0
        num_holes = 0
        transpose = zip(*stationary_board)
        for col in transpose:
            row = 0
            while row < c.ROWS and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes
    
    
    def get_bumpiness_and_height(self, board=None):
        board = self.board if board == None else board
        stationary_board = [row[:] for row in board]
        for (row, col) in self.current_piece.positions:
            stationary_board[row][col] = 0
        board = np.array(stationary_board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), c.ROWS)
        heights = c.ROWS - invert_heights
        total_height = np.sum(heights)
        diffs = np.abs(heights[:-1] - heights[1:])
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height
    

    def update_current_piece(self, piece):
        self.current_piece = piece
        self.modify_board(self.current_piece.positions, 1)


    def move_in_board(self, new_pos):
        self.modify_board(self.current_piece.positions, 0)
        self.modify_board(new_pos, 1)


    def modify_board(self, positions, value):
        for (row, col) in positions:
            self.board[row][col] = value
        

    def set_piece(self):
        for (row, col) in self.current_piece.positions:
            self.color_board[(row, col)] = self.current_piece.color
        self.latest_position = self.current_piece.positions
        self.current_piece = Piece(c.O, (0, 0, 0)) 
        self.reward = self.rows_cleared ** 2 * 10 + 1


    def empty_row(self, row_index):
        self.rows_cleared += 1
        for column in range(c.COLUMNS):
            self.board[row_index][column] = 0
        for pos, color in self.color_board.items():
            if pos[0] == row_index: self.color_board[pos] = (0, 0, 0)


    def shift_down(self, emptied_row_index):
        for index in range(emptied_row_index, 0, -1):
            self.board[index] = copy.deepcopy(self.board[index-1])
        

    def empty_places_below(self):
        empty_places = 0
        for (row, col) in self.current_piece.positions:
            row_index = row
            count = 0
            while row_index < c.ROWS-1 and self.board[row_index + 1][col] == 0:
                count += 1; row_index += 1
            empty_places += count
        return empty_places


    def increment_score(self, amount=1):
        self.score += amount


    def empty_board(self):
        self.board = [[0] * c.COLUMNS for _ in range(c.ROWS)]
        self.color_board = {(x, y): (0, 0, 0) for x in range(20) for y in range(10)}


    def has_active_piece(self):
        return self.current_piece.color != (0, 0, 0) 

        




    
