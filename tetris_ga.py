"""
SRN 1: PES2UG23CS247
SRN 2: PES2UG23CS234
"""

import sys
import time
import random
import numpy as np
import pygame
from copy import deepcopy

# ---------------- CONFIG ----------------
GRID_W, GRID_H = 10, 20
CELL = 25
SIDE = 200
W, H = GRID_W * CELL + SIDE, GRID_H * CELL
FPS = 60

# Genetic Algorithm
POP_SIZE = 20
GENERATIONS = 30
TOURNAMENT = 3
CROSSOVER_RATE = 0.8
MUT_RATE = 0.15
MUT_STD = 0.3
ELITISM = 2
MAX_STEPS = 1000

# -------------- PIECES -------------------
PIECES = {
    'I': [[[1,1,1,1]], [[1],[1],[1],[1]]],
    'O': [[[1,1],[1,1]]],
    'T': [[[0,1,0],[1,1,1]], [[1,0],[1,1],[1,0]], [[1,1,1],[0,1,0]], [[0,1],[1,1],[0,1]]],
    'S': [[[0,1,1],[1,1,0]], [[1,0],[1,1],[0,1]]],
    'Z': [[[1,1,0],[0,1,1]], [[0,1],[1,1],[1,0]]],
    'J': [[[1,0,0],[1,1,1]], [[1,1],[1,0],[1,0]], [[1,1,1],[0,0,1]], [[0,1],[0,1],[1,1]]],
    'L': [[[0,0,1],[1,1,1]], [[1,0],[1,0],[1,1]], [[1,1,1],[1,0,0]], [[1,1],[0,1],[0,1]]],
}
PIECE_NAMES = list(PIECES.keys())

# -------------- ENVIRONMENT --------------
class Tetris:
    def __init__(self, w=GRID_W, h=GRID_H):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        self.grid = [[0]*self.w for _ in range(self.h)]
        self.score, self.lines, self.done = 0, 0, False
        self.spawn_piece()
        self.steps = 0
        return self.get_state()

    def spawn_piece(self):
        self.name = random.choice(PIECE_NAMES)
        self.rots = PIECES[self.name]
        self.rot_i = 0
        self.piece = self.rots[0]
        self.ph, self.pw = len(self.piece), len(self.piece[0])
        self.x, self.y = (self.w - self.pw)//2, 0
        if self.collides(self.x, self.y, self.piece):
            self.done = True

    def collides(self, x, y, shape):
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    yy, xx = y + r, x + c
                    if yy < 0 or yy >= self.h or xx < 0 or xx >= self.w:
                        return True
                    if self.grid[yy][xx]:
                        return True
        return False

    def lock_piece(self):
        for r in range(len(self.piece)):
            for c in range(len(self.piece[0])):
                if self.piece[r][c]:
                    self.grid[self.y + r][self.x + c] = 1
        cleared = self.clear_lines()
        self.score += (cleared ** 2) * 100
        self.lines += cleared
        self.spawn_piece()

    def clear_lines(self):
        new = [row for row in self.grid if not all(row)]
        cleared = self.h - len(new)
        while len(new) < self.h:
            new.insert(0, [0]*self.w)
        self.grid = new
        return cleared

    def move(self, dx):
        if not self.collides(self.x + dx, self.y, self.piece):
            self.x += dx

    def rotate(self):
        nxt = (self.rot_i + 1) % len(self.rots)
        newp = self.rots[nxt]
        if not self.collides(self.x, self.y, newp):
            self.rot_i = nxt
            self.piece = newp
            self.ph, self.pw = len(newp), len(newp[0])

    def drop(self):
        while not self.collides(self.x, self.y + 1, self.piece):
            self.y += 1
        self.lock_piece()

    def step(self, action):
        if self.done: return self.get_state(), 0, True
        self.steps += 1
        if action == 'left': self.move(-1)
        elif action == 'right': self.move(1)
        elif action == 'rotate': self.rotate()
        elif action == 'drop': self.drop()
        if not self.collides(self.x, self.y + 1, self.piece):
            self.y += 1
        else:
            self.lock_piece()
        if self.steps >= MAX_STEPS:
            self.done = True
        return self.get_state(), self.score, self.done

    def get_state(self):
        g = [r[:] for r in self.grid]
        for r in range(len(self.piece)):
            for c in range(len(self.piece[0])):
                if self.piece[r][c]:
                    yy, xx = self.y + r, self.x + c
                    if 0 <= yy < self.h and 0 <= xx < self.w:
                        g[yy][xx] = 1
        return {'grid': g, 'score': self.score, 'lines': self.lines, 'done': self.done}

    # Features
    def heights(self):
        h = [0]*self.w
        for c in range(self.w):
            for r in range(self.h):
                if self.grid[r][c]:
                    h[c] = self.h - r
                    break
        return h
    def holes(self):
        holes=0
        for c in range(self.w):
            seen=False
            for r in range(self.h):
                if self.grid[r][c]: seen=True
                elif seen: holes+=1
        return holes
    def agg_height(self): return sum(self.heights())
    def bumpiness(self):
        h = self.heights()
        return sum(abs(h[i]-h[i+1]) for i in range(len(h)-1))
    def full_lines(self): return sum(1 for r in self.grid if all(r))

# ------------- AGENT --------------------
class Agent:
    def __init__(self, w=None):
        self.w = np.random.randn(5)*0.1 if w is None else np.array(w)

    def act(self, env:Tetris):
        best_val, best = -1e9, 'noop'
        for ridx in range(len(env.rots)):
            shape = env.rots[ridx]
            for x in range(env.w - len(shape[0]) + 1):
                sim = deepcopy(env)
                sim.rot_i = ridx
                sim.piece = shape
                sim.x, sim.y = x, 0
                if sim.collides(sim.x, sim.y, sim.piece): continue
                while not sim.collides(sim.x, sim.y+1, sim.piece): sim.y+=1
                sim.lock_piece()
                f = np.array([
                    sim.full_lines(),
                    -sim.agg_height(),
                    -sim.holes(),
                    -sim.bumpiness(),
                    1.0
                ])
                val = np.dot(self.w, f)
                if val > best_val:
                    best_val, best = val, (ridx, x)
        if best == 'noop': return 'noop'
        ridx, x = best
        if ridx != env.rot_i: return 'rotate'
        if x < env.x: return 'left'
        if x > env.x: return 'right'
        return 'drop'

# ------------- GA -----------------------
def eval_agent(a:Agent, episodes=1):
    total = 0
    for _ in range(episodes):
        e = Tetris()
        e.reset()
        steps = 0
        while not e.done and steps < MAX_STEPS:
            e.step(a.act(e))
            steps+=1
        total += e.score + e.lines*1000
    return total/episodes

def tournament(pop, fits):
    res=[]
    for _ in range(len(pop)):
        ids=random.sample(range(len(pop)),TOURNAMENT)
        best=max(ids,key=lambda i:fits[i])
        res.append(deepcopy(pop[best]))
    return res

def cross(a,b):
    if random.random()>CROSSOVER_RATE: return deepcopy(a),deepcopy(b)
    L=len(a.w); pt=random.randint(1,L-1)
    return Agent(np.concatenate([a.w[:pt],b.w[pt:]])), Agent(np.concatenate([b.w[:pt],a.w[pt:]]))

def mutate(a):
    for i in range(len(a.w)):
        if random.random()<MUT_RATE: a.w[i]+=np.random.normal(0,MUT_STD)
    return a

def evolve():
    pop=[Agent() for _ in range(POP_SIZE)]
    for g in range(GENERATIONS):
        fits=[eval_agent(a) for a in pop]
        best_i=int(np.argmax(fits))
        print(f"Gen {g} best={fits[best_i]:.1f}")
        elite=[deepcopy(pop[i]) for i in np.argsort(fits)[-ELITISM:][::-1]]
        sel=tournament(pop,fits)
        newp=list(elite)
        while len(newp)<POP_SIZE:
            a,b=random.choice(sel),random.choice(sel)
            c1,c2=cross(a,b)
            newp.append(mutate(c1))
            if len(newp)<POP_SIZE:newp.append(mutate(c2))
        pop=newp
    best_i=int(np.argmax([eval_agent(a) for a in pop]))
    return pop[best_i]

# ------------- RENDER -------------------
pygame.init()
screen=pygame.display.set_mode((W,H))
pygame.display.set_caption("Tetris GA")
clock=pygame.time.Clock()
FONT=pygame.font.SysFont("Arial",16)

def draw(env:Tetris,gen=0,score=0):
    screen.fill((0,0,0))
    for r in range(env.h):
        for c in range(env.w):
            rect=pygame.Rect(c*CELL,r*CELL,CELL,CELL)
            if env.grid[r][c]:
                pygame.draw.rect(screen,(200,200,200),rect)
            pygame.draw.rect(screen,(50,50,50),rect,1)
    for r in range(len(env.piece)):
        for c in range(len(env.piece[0])):
            if env.piece[r][c]:
                rr,cc=env.y+r,env.x+c
                rect=pygame.Rect(cc*CELL,rr*CELL,CELL,CELL)
                pygame.draw.rect(screen,(180,180,220),rect)
    sx=env.w*CELL+10
    screen.blit(FONT.render(f"Lines: {env.lines}",True,(255,255,255)),(sx,20))
    screen.blit(FONT.render(f"Score: {env.score}",True,(255,255,255)),(sx,50))
    screen.blit(FONT.render(f"Gen: {gen}",True,(255,255,255)),(sx,80))

# ------------- MAIN ---------------------
def manual():
    e = Tetris()
    e.reset()
    fall_time = 0
    fall_speed = 0.5  # seconds per drop
    last_drop = time.time()
    run = True

    while run:
        now = time.time()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                run = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_LEFT:
                    e.step('left')
                elif ev.key == pygame.K_RIGHT:
                    e.step('right')
                elif ev.key == pygame.K_UP:
                    e.step('rotate')
                elif ev.key == pygame.K_SPACE:
                    e.step('drop')
                elif ev.key == pygame.K_ESCAPE:
                    run = False

        # make pieces fall over time
        if now - last_drop > fall_speed:
            e.step('noop')
            last_drop = now

        draw(e)
        pygame.display.flip()
        clock.tick(FPS)

        if e.done:
            time.sleep(1)
            e.reset()

    pygame.quit()


def auto():
    print("Training agents... please wait.")
    best=evolve()
    e=Tetris(); e.reset()
    while True:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:pygame.quit();sys.exit()
            if ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE:pygame.quit();sys.exit()
        e.step(best.act(e))
        draw(e)
        pygame.display.flip()
        clock.tick(10)
        if e.done:
            time.sleep(1)
            e=Tetris(); e.reset()

if __name__=="__main__":
    print("Tetris + GA")
    print("1: Play manually")
    print("2: Train & watch best AI")
    ch=input("Choose (1/2): ").strip()
    if ch=='1': manual()
    else: auto()
