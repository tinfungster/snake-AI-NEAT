import os
import pickle
import numpy as np
from snake import Game
import neat
import pygame

runs_per_net = 5

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for runs in range(runs_per_net):
        game = Game(20,20)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while True:
            inputs = game.get_normalized_state()
            action = net.activate(inputs)

            # Apply action to the simulated snake
            valid = game.step(np.argmax(action))

            # Stop if the network fails to keep the snake within the boundaries or hits itself.
            # The per-run fitness is the number of pills eaten
            if not valid:
                break
           
            fitness = game.fitness


        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def main(): 
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate,10000)
    print(f"WINNER: {winner}")
    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    
if __name__ == '__main__':
    main()