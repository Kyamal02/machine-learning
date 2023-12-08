import pygame
import sys

# Инициализация Pygame
pygame.init()

# Устанавливаем размеры окна
width, height = 1200, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Рисовалка в Pygame")

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Флаги для рисования
drawing = False
last_pos = None

# Заливаем фон белым цветом
screen.fill(WHITE)

# Основной цикл программы
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                pygame.draw.line(screen, BLACK, last_pos, current_pos, 20)
                last_pos = current_pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                pygame.image.save(screen, 'saved_drawing.png')

    pygame.display.flip()
