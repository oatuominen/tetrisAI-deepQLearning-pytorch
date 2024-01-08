from turtle import pos, position
from numpy import full
import sys
import copy
import pygame
import torch as T
import random as r
import time
import constants as c
from piece import Piece
from state import State
from agent import Agent

class Game:

    def __init__(self, app):
        self.app = app
        self.state = State()
        self.agent = Agent(learning_rate=0.001, discount_factor=0.99, epsilon=0.3)
        self.piece_set = False
        self.running = False
        

    def update(self, event, automatic=False):
        if event == "down":
            self.app.row_cleared_reward = 0.0
            if not self.move("down"): self.set_piece()
            elif not automatic: self.state.increment_score()
        elif event == "left":
            self.move("left")
        elif event == "right":
            self.move("right")
        elif event == "up":
            self.move("up")
        elif event == "d":
            while self.state.has_active_piece():
                self.update("down")


    def one_piece_journey(self, train=True):
        self.state.reward = 0.0 #?
        final_pos = []
        final_positions = self.generate_positions()
        
        if self.agent.random_journey() and train:

            while final_positions:
                destination = r.choice(final_positions)
                board = self.state.stat_board()
                if self.can_move(destination, board):
                    final_pos = destination
                    self.move_to(destination)
                    self.set_piece()
                    return final_pos, self.state.reward, self.state.to_id(), False
                final_positions.remove(destination)
            return [], -1, self.state.to_id(), True

        else:
            possible_states = []
            board = self.state.stat_board()
            max_q_value = -sys.float_info.max

            for pos in final_positions:
                for (row, col) in pos:
                    board[row][col] = 1
                possible_states.append((pos, board))
                board = self.state.stat_board()
            
            if possible_states == []: 
                return [], -1, self.state.to_id(), True

            for (pos, board_) in possible_states:
                state_tensor = T.tensor([self.state.to_id(board_)], dtype=T.float32).to(self.agent.dqn.device)
                q_estimate = self.agent.dqn.forward(state_tensor)
                if q_estimate.item() >= max_q_value:
                    max_q_value = q_estimate.item()
                    final_pos = pos

            self.move_to(final_pos)
            self.set_piece()
            return final_pos, self.state.reward, self.state.to_id(), False



    def generate_positions(self):
        final_positions = []
        board = self.state.stat_board()
        rotation = -1
        for rotation in range(self.state.current_piece.num_rotations):
            rotated = self.state.current_piece.movement("up", rotation=rotation)
            rotation += 1
            left = self.state.current_piece.movement("left", initial_position=rotated)
            while self.can_move(left, board):
                left = self.state.current_piece.movement("left", initial_position=left)
            left = self.state.current_piece.movement("right", initial_position=left)
            for column in range(c.COLUMNS):
                right = left
                for _ in range(column):
                    right = self.state.current_piece.movement("right", initial_position=right)
                if not self.can_move(right, board): 
                    continue
                drop = self.state.current_piece.movement("down", initial_position=right)
                while self.can_move(drop, board):
                    drop = self.state.current_piece.movement("down", initial_position=drop)
                drop = self.state.current_piece.movement("fix", initial_position=drop)
                final_positions.append(drop)
        return final_positions


    def move(self, direction):
        new_pos = self.state.current_piece.movement(direction)
        if self.can_move(new_pos, self.state.stat_board()):
            self.move_to(new_pos, dir=direction)
            return True
        else: return False


    def move_to(self, new_pos, dir=None):
        self.state.move_in_board(new_pos)
        old_pos = self.state.current_piece.positions
        self.erase(old_pos)
        if dir is not None:
            self.state.current_piece.movement(dir, move=True)
        self.draw_piece(new_pos, self.state.current_piece.color)
        self.state.current_piece.positions = new_pos


    def can_move(self, position, board):
        return all(
            row >= 0 and row < c.ROWS and 
            col >= 0 and col < c.COLUMNS and
            (board[row][col] != 1)
            for (row, col) in position)


    def draw_piece(self, positions, color):
        pixel_locations = self.grid_to_pixel(positions)
        for (x, y) in pixel_locations:
            pygame.draw.rect(self.app.screen, color,
                            (x, y, c.BLOCK_SIZE, c.BLOCK_SIZE))
            
            
    def grid_to_pixel(self, gridPos): 
        return [(col * c.BLOCK_SIZE, row* c.BLOCK_SIZE) for (row, col) in gridPos]
    

    def erase(self, old_pos):
        self.draw_piece(old_pos, c.BACKGROUND)


    def new_piece(self):
        index = r.randint(0, 6)
        shape = c.PIECES[index]
        color = c.COLORS[index]
        piece = Piece(shape, color)
        piece.initialize()
        if not self.can_move(piece.positions, self.state.stat_board()):
            self.state.reward -= 100
            self.running = False
            return
        else: 
            self.state.update_current_piece(piece)
        self.draw_piece(piece.positions, piece.color)


    def set_piece(self):
        self.check_loss()
        self.state.set_piece()
        self.check_full_rows()
        

    def check_full_rows(self):
        full_rows = []
        for index, row in enumerate(self.state.board):
            if all(elem == 1 for elem in row):
                full_rows.append(index)
        if full_rows:
            self.add_score(len(full_rows))
            sorted_full_rows = sorted(full_rows)
            for row in sorted_full_rows:
                self.empty_row(row)
        return len(full_rows)


    def empty_row(self, row):
        self.state.empty_row(row)
        row_positions = [(row, col) for col in range(c.COLUMNS)]
        self.draw_piece(row_positions, c.BACKGROUND)
        self.shift_down(row)
        self.check_level()


    def add_score(self, rows_cleared):
        score_multipliers = {1: 40, 2: 100, 3: 300, 4: 1200}
        self.state.increment_score(score_multipliers[rows_cleared] * (self.state.level + 1))
        self.app.row_cleared_reward = rows_cleared ** 2 * 10 + 1


    def check_level(self):
        if self.state.rows_cleared % 10 == 0:
            self.state.level += 1
            if self.state.level < 10:
                self.app.gravity -= 50
            elif self.state.level in {13, 16, 19, 29}:
                self.app.gravity -= 50


    def shift_down(self, row):
        shift_rows = [(pos, color) for (pos, color) in self.state.color_board.items() if pos[0] < row]
        shift_rows = shift_rows[::-1]
        for (pos, color) in shift_rows:
            shifted_pos = (pos[0] + 1, pos[1])
            self.state.color_board[pos]= (0, 0, 0)
            self.state.color_board[shifted_pos] = color
            self.draw_piece([pos], (0, 0, 0))
            self.draw_piece([shifted_pos], color)
        self.state.shift_down(row)


    def check_loss(self):
        for (row, col) in self.state.current_piece.positions:
            if row == 0:
                self.state.reward -= 100
                self.running = False
                

    def empty_grid(self):
        all_positions = [(row, col) for row in range(0, c.ROWS) for col in range(0, c.COLUMNS)]
        self.draw_piece(all_positions,(0, 0, 0))
        self.state.empty_board()
        self.state.current_piece = Piece(c.O, (0, 0, 0))


    def restart(self):
        self.start()
        self.new_piece()


    def start(self):
        self.empty_grid()
        self.state.level = 0
        self.state.score = 0
        self.state.rows_cleared = 0
        self.state.reward = 0.0
        self.running = True


    def quit_game(self):
        self.empty_grid()
        self.running = False



    

