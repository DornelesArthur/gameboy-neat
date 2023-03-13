#!/usr/bin/python                                                                                                                                                                        

import os
import socket
import threading
import neat
import pickle
import time

HEADER = 64
PORT = 37259
TRAINERS_NUMBER = 5
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'latin1'

class Manager():
    def __init__(self):
        print("[CREATING] creating socket")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((SERVER, PORT))
        print (f"[STARTED] Server started in {SERVER}!")
        self.server.listen(TRAINERS_NUMBER)
        print ("[LISTENING]")

    def handle_trainer(self,conn,addr,genome,genome_id):
        genome_bytes = pickle.dumps(genome)
        print(f"[GENOME] Preparing genome")
        genome_size = len(genome_bytes)
        genome_size = str(genome_size).encode(FORMAT)
        genome_size += b' ' * (HEADER- len(genome_size))
        print(f"[GENOME] Sending genome size ={genome_size}")
        conn.send(genome_size)
        print(f"[GENOME] Sending {type(genome_bytes)}")
        conn.send(genome_bytes)
        print(f"[FITNESS] Fitness size ={HEADER}")
        fitness = int(conn.recv(HEADER))
        print(f"[FITNESS] {fitness} - {id}")
        genome.fitness = fitness
        print(f"[CLOSING] closing socket connection")
        conn.close()

    def eval_genomes(self,genomes, config):
        print("[EVALUATING]")
        for i, (genome_id, genome) in enumerate(genomes):
            conn, addr = self.server.accept()
            print (f"{conn} {addr}")
            thread = threading.Thread(target=self.handle_trainer,args=(conn,addr,genome,genome_id))
            thread.start()
        while (threading.active_count() != 1):
            print(f"[WAIT] Active Connections: {threading.active_count()-1}")
            time.sleep(1)

    def run_neat(self,config):
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(1))

        best_genome = pop.run(self.eval_genomes, 12)
        with open("best.pickle", "wb") as f:
            pickle.dump(best_genome,f)

if __name__ == "__main__":
    print(f'[STARTING]')
    print(f'[CONFIG] Loading config file')
    local_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(local_dir, "config.txt")

    print(f'[CONFIG] Getting config info')
    config= neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    manager = Manager()
    print(f'[NEAT] Initializing')
    manager.run_neat(config)