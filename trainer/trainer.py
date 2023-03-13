#!/usr/bin/python

import os
import sys
import time
from pyboy import PyBoy, WindowEvent
import numpy as np
import neat
import pickle
import socket

HEADER = 32
PORT = 37259
FORMAT = 'latin1'
SERVER = "127.0.1.1"
ADDR = (SERVER, PORT)

class SuperMarioLandGame:
    def __init__(self):
        file_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.insert(0, file_path + "/..")
        quiet = "--quiet" in sys.argv
        self.pyboy = PyBoy('SuperMarioLand.gb', window_type="headless" if quiet else "SDL2", window_scale=3, debug=not quiet, game_wrapper=True, disable_renderer=True)
        assert self.pyboy.cartridge_title() == "SUPER MARIOLAN"
        self.env = self.pyboy.game_wrapper()
        self.pyboy.set_emulation_speed(0)

    def start(self, config):
        print(f'[GAME] Starting')
        self.env.start_game()
        assert self.env.score == 0
        assert self.env.lives_left == 2
        assert self.env.time_left == 400
        assert self.env.world == (1, 1)
        assert self.env.fitness == 0
        self.run(config)
    
    def run(self, config):
        print('[RUNNING]')
        while True:
            try:
                print(f"[TRYING] Conecting with the server")
                client_thread_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_thread_server.connect(ADDR)

                print(f"[GENOMA] Waiting for genoma")
                genome_size = int(client_thread_server.recv(HEADER).decode(FORMAT))

                print(f"[GENOMA] Genoma size received: {genome_size}")
                genome_bytes = b""
                i=0
                while True:
                    part = client_thread_server.recv(genome_size)
                    i+=1
                    genome_bytes += part
                    print(f"Part = {i} - {len(genome_bytes)}")
                    if len(genome_bytes) >= genome_size:
                        break
                print(f"[GENOMA] Actual size: {len(genome_bytes.strip())}")

                genome_pickle = pickle.loads(genome_bytes.strip())
                print(f"[RECEIVE] {type(genome_pickle)}")

                fitness = self.train_ai(genome_pickle, config)
                message = str(fitness).encode(FORMAT)
                print(f"[SENDING] Fitness {message}")
                client_thread_server.send(message)
                print(f"[SENDED]")
                print(f"[CLOSING]")
                client_thread_server.close()
                print(f"[CLOSED]")
            except socket.error as exc:
                print(f"[ERROR] Error when tried to conect socket: {exc}")
                time.sleep(0.5)
            

    def train_ai(self, genome, config):
        # print(type(genome))
        # with open("genome.pickle", "wb") as f:
        #     pickle.dump(genome,f)
        print('[NEURAL NETWORK] Creating')
        network = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        i = 0
        old_score = 0
        old_pos = self.env.level_progress
        # old_lives = 2
        print('[TRAINING]')
        self.env.reset_game()
        while run:
            if self.env.game_over():
                run = False
                # genome.fitness = ((self.env.score/1000)+(self.env.world[0]*100000)+(self.env.world[1]*10000)+(self.env.level_progress/10))/4
                # genome.fitness = self.env.fitness
            else:
                i+=1
                # print(i)
                if i == 300:
                    if self.env.score == old_score and old_pos >= self.env.level_progress:
                        # print(f"{i} - if 1 - {self.env.score} - {old_score}")
                        run = False
                        print
                        # genome.fitness = ((self.env.score/1000)+(self.env.world[0]*100000)+(self.env.world[1]*10000)+(self.env.level_progress/10))/4
                        # genome.fitness = self.env.fitness
                elif i > 300:
                    # print(f"{i} - if 2 - {self.env.score} - {old_score}")
                    old_score = self.env.score
                    old_pos = self.env.level_progress
                    i = 0
                # coins = self.pyboy.get_memory_value(0x9829)*10 + self.pyboy.get_memory_value(0x982A)
                # time = self.pyboy.get_memory_value(0x9831)*100 + self.pyboy.get_memory_value(0x9832)*10 + self.pyboy.get_memory_value(0x9833)
                # input_array = np.r_[np.array(self.pyboy.screen_image().convert('L').getdata()), self.pyboy.get_memory_value(0x9806), coins, time]
                input_array = np.array(self.pyboy.screen_image().convert('L').getdata())
                output = network.activate((input_array))

                self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP) if output[0] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN) if output[1] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT) if output[2] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT) if output[3] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A) if output[4] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B) if output[5] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
                # self.pyboy.send_input(WindowEvent.PRESS_BUTTON_SELECT) if output[6] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_SELECT)
                # self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START) if output[7] > 0.5 else self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                self.pyboy.tick()
        return self.env.fitness

if __name__ == "__main__":
    print(f'[STARTING]')
    print(f'[CONFIG] Loading config file')
    local_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(local_dir, "config.txt")

    print(f'[CONFIG] Getting config info')
    config= neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    print(f'[GAME] Initializing')
    game = SuperMarioLandGame()
    game.start(config)