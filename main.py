
import pygame
import sys
import time
import constants as c
from game import Game


class App:

    def __init__(self):
        self.game = Game(self)
        pygame.init()
        pygame.display.set_caption("TetrisAI")
        self.screen = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
        self.clock = pygame.time.Clock()
        self.fall_timer = 0  
        self.action_timer = 0
        self.gravity = 1000
        self.action_speed = 200
        self.num_games = 5000
        

    def train_agent(self):
        game_num = 0
        while game_num < self.num_games:
            if game_num % 10 == 0: self.game.agent.dqn.save_dqn()
            self.game.start()
            while self.game.running:
                self.draw()
                if not self.game.state.has_active_piece():
                    self.game.new_piece()
                    self.game.row_cleared_reward = 0.0
                    initial_state = self.game.state.to_id()
                    final_position, reward, new_state, done = self.game.one_piece_journey() 
                    reward += self.game.row_cleared_reward
                    if done: 
                        reward -= 100
                        self.game.running=False
                    self.game.agent.store_transition(initial_state, final_position, new_state, reward, done)
                    self.game.agent.learn()
                self.react_to_events(interactive=False)
                pygame.display.flip()
                self.clock.tick(60)
            game_num += 1


    def view_agent_play(self):
        game_num = 0
        while game_num < self.num_games:
            self.game.start()
            while self.game.running:
                self.draw()
                if not self.game.state.has_active_piece():
                    if self.time_to_act():
                        self.game.new_piece()
                        action, reward, new_state, done = self.game.one_piece_journey(train=False)
                        self.game.check_full_rows
                        if done == True: 
                            reward -= 100
                            self.game.running=False
                    else: continue
                    self.action_timer = pygame.time.get_ticks()
                self.react_to_events(interactive=False)
                pygame.display.flip()
                self.clock.tick(60)
                time.sleep(0.1)
            game_num += 1


    def play_manual(self):
        self.game.running = True
        while True:
            self.draw()
            self.make_buttons()
            if self.game.running:
                if self.game.state.has_active_piece():
                    self.auto_fall()
                else: self.game.new_piece()
            self.react_to_events()
            pygame.display.flip()
            self.clock.tick(60)


    def auto_fall(self):
        if self.time_to_fall():
            self.game.update("down", automatic=True)
            self.fall_timer = pygame.time.get_ticks()
            return True
        else: return False


    def time_to_fall(self):
        current_time = pygame.time.get_ticks()
        return current_time - self.fall_timer >= self.gravity
        

    def time_to_act(self):
        current_time = pygame.time.get_ticks()
        return current_time - self.action_timer >= self.action_speed


    def react_to_events(self, interactive=True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif interactive:
                if event.type == pygame.KEYDOWN:
                    self.game.update(pygame.key.name(event.key))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.start_button.collidepoint(event.pos):
                        self.game.restart()
                    elif self.restart_button.collidepoint(event.pos):
                        self.game.restart()
                    elif self.quit_button.collidepoint(event.pos):
                        self.game.quit_game()

    def draw(self):
        self.display_text(f'Score: {self.game.state.score}', c.SCORE_Y)
        self.display_text(f'Level: {self.game.state.level}', c.LEVEL_Y)

    def make_buttons(self):
        self.start_button = self.display_button(self.display_text("START", c.START_BUTTON_Y), c.START_BUTTON_Y)
        self.restart_button = self.display_button(self.display_text("RESTART", c.RESTART_BUTTON_Y), c.RESTART_BUTTON_Y)
        self.quit_button = self.display_button(self.display_text("QUIT", c.QUIT_BUTTON_Y), c.QUIT_BUTTON_Y)

    def draw_grid(self):
        for x in range(0, c.GRID_WIDTH+c.BLOCK_SIZE, c.BLOCK_SIZE):
            pygame.draw.line(self.screen, c.GRID_COLOR, (x, 0), (x, c.GRID_HEIGHT))
        for y in range(0, c.GRID_HEIGHT, c.BLOCK_SIZE):
            pygame.draw.line(self.screen, c.GRID_COLOR, (0, y), (c.GRID_WIDTH, y))

    def display_button(self, text, y_loc):
        button = pygame.Rect(c.BUTTON_X, y_loc, c.BUTTON_WIDTH, c.BUTTON_HEIGHT)
        pygame.draw.rect(self.screen, (255, 0, 0), (c.BUTTON_X, y_loc, c.BUTTON_WIDTH, c.BUTTON_HEIGHT))
        text_rect = text.get_rect(center=button.center)
        self.screen.blit(text, text_rect)
        return button
    
    def display_text(self, string, y_loc):
        text_location = (c.BUTTON_X, y_loc)
        text_area = pygame.Surface(text_location)
        text_area.fill((0, 0, 0))
        self.screen.blit(text_area, text_location)
        font = pygame.font.Font(None, c.FONT_SIZE)
        text = font.render(string, True, (255, 255, 255))
        self.screen.blit(text, text_location)
        return text
        
app = App()
app.view_agent_play()



