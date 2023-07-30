import numpy
from deap import base
from deap import tools
from deap import creator
from deap import algorithms

import random

import matplotlib.pyplot as plt
import seaborn as sns

# the following program is equivalent to the "OneMaxLong" one but uses built-in algorithms

IND_LENGTH = 100

POPULATION_SIZE = 200
MAX_GENERATIONS = 50

P_MUTATION = 0.1
P_CROSSOVER = 0.9

toolbox = base.Toolbox()

toolbox.register("zeroOrOne", random.randint, 0, 1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=toolbox.FitnessMax)

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, IND_LENGTH)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def oneMaxFitness(Individual):
    return sum(Individual),


toolbox.register("evaluate", oneMaxFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / IND_LENGTH)


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats, verbose=True)
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()
