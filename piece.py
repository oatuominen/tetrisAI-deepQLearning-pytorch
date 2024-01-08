
import numpy as np
import constants as c

class Piece:

    def __init__(self, shapes, color):
        self.shapes = shapes
        self.shape_index = 0
        self.format = shapes[self.shape_index]
        self.color = color
        self.rotation = 0
        self.positions = []
        self.x = 4
        self.y = 0
        self.active = True
        self.num_rotations = len(self.shapes)

    def initialize(self):
        for (i, row) in enumerate(self.format):
            for (j, square) in enumerate(list(row)):
                if square == "0":
                    self.positions.append((i + self.y, j + self.x))
   
    def updated_format(self, format):
        new_pos = []
        for (i, row) in enumerate(format):
            for (j, square) in enumerate(list(row)):
                if square == "0":
                    new_pos.append((i + self.y, j + self.x))
        return new_pos



    def movement(self, direction, move=False, initial_position=[], rotation=None):
        positions = self.positions if len(initial_position)==0 else initial_position
        if direction == "down":
            new_pos = [(row + 1, col) for (row, col) in positions]
            if move: 
                self.y += 1
                self.positions = new_pos
            else: return new_pos

        elif direction == "left":
            new_pos = [(row, col-1) for (row, col) in positions]
            if move: 
                self.x -= 1
                self.positions = new_pos
            else: return new_pos

        elif direction == "right":
            new_pos = [(row, col+1) for (row, col) in positions]
            if move: 
                self.x += 1
                self.positions = new_pos
            else: return new_pos

        elif direction == "up":
            index = rotation if rotation is not None else self.shape_index
            next_index = (index + 1) % len(self.shapes)
            shape = self.shapes[next_index]
            new_pos = self.updated_format(shape)
            if move:
                self.positions = new_pos
                self.shape_index = next_index
                self.format = self.shapes[self.shape_index]
            else: return new_pos
        
        elif direction == "fix":
            return [(row - 1, col) for (row, col) in positions]

    



#delete



    def rotate(self, positions, shape_index):
        next_shape = self.shapes[shape_index % len(self.shapes)]
        return self.updated_format(next_shape)
    
    def fix_up(self, positions):
        return [(row - 1, col) for (row, col) in positions]










