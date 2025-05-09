import numpy as np
import pygame
import os
import assets
import configs
from objects.background import Background
from objects.bird import Bird
from objects.column import Column
from objects.floor import Floor
from objects.score import Score
from objects.gameover_message import GameOverMessage
from objects.gamestart_message import GameStartMessage


# class for the game environment
class FlappyBirdGame:
    def __init__(self, headless=False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # Disable window creation

        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            pygame.mixer.init = lambda *args, **kwargs: None
            pygame.mixer.Sound = lambda *args, **kwargs: None
        pygame.init()
        self.screen = None if headless else pygame.display.set_mode((configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        assets.load_sprites()
        assets.load_audios()

        self.column_create_event = pygame.USEREVENT
        self.sprites = pygame.sprite.LayeredUpdates()

        # Game state flags
        self.running = True
        self.gameover = False
        self.gamestarted = False
        self.done = False

        self._create_sprites()

    def _create_sprites(self):
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)

        self.bird = Bird(self.sprites)
        self.score = Score(self.sprites)


    def step(self, action):
        reward = 0.1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.done = True
            if event.type == self.column_create_event:
                Column(self.sprites)
        if action == 1:
            self.bird.jump()
            reward -= 0.1

        if self.gamestarted and not self.done:
            self.sprites.update()

        if self.bird.check_collision(self.sprites):
            self.done = True
            pygame.time.set_timer(self.column_create_event, 0)
            reward = -10
        else:
            next_col = None
            for sprite in self.sprites:
                if isinstance(sprite, Column) and sprite.rect.right > self.bird.rect.left:
                    next_col = sprite
                    break
            if next_col:
                pipe_center_y = next_col.rect.y + next_col.gap / 2
                bird_center_y = self.bird.rect.centery
                alignment_error = abs(pipe_center_y - bird_center_y) / next_col.gap
                alignment_bonus = max(0, 1.0 - alignment_error)
                reward += alignment_bonus * 0.3

        for sprite in self.sprites:
            if isinstance(sprite, Column) and sprite.is_passed():
                self.score.value += 1
                reward += 1
        return self._get_state(), reward, self.done, {}

    def render(self):
        if self.screen:
            self.screen.fill(0)
            self.sprites.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(configs.FPS)

    def _get_state(self):
        bird_y = self.bird.rect.y / configs.SCREEN_HEIGHT
        bird_flap = np.clip(self.bird.flap, -10, 10) / 10.0

        next_col = None
        for sprite in self.sprites:
            if isinstance(sprite, Column) and sprite.rect.right > self.bird.rect.left:
                next_col = sprite
                break

        if next_col is not None:
            distance_x = (next_col.rect.x - self.bird.rect.x) / configs.SCREEN_WIDTH
            pipe_center_y = next_col.rect.y + next_col.gap / 2
            bird_center_y = self.bird.rect.centery
            distance_y = (pipe_center_y - bird_center_y) / configs.SCREEN_HEIGHT
        else:
            distance_x = 1
            distance_y = 0

        state = np.array([
            bird_y,
            bird_flap,
            distance_x,
            distance_y
        ], dtype=np.float32)

        return state

    def reset(self):
        self.sprites.empty()
        self._create_sprites()
        self.gameover = False
        self.gamestarted = True
        self.done = False
        pygame.time.set_timer(self.column_create_event, 1500)
        pygame.event.clear()
        return self._get_state()
    def close(self):
        pygame.quit()
        self.running = False
