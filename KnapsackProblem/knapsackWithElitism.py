from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import knapsack

# Instantiating an instance of the knapsack problem, the class is contained in the "knapsack.py" file in this directory,
# and it's from RosettaCode.org
knapsack = knapsack.Knapsack01Problem()

# Problem Constants
POPULATION_SIZE = 70
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 70
HALL_OF_FAME_SIZE = 1

toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(knapsack))
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def knapsackValue(Individual):
    return knapsack.getValue(Individual),


toolbox.register("evaluate", knapsackValue)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(knapsack))


def main():

    population = toolbox.populationCreator(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    best = hof.items[0]
    print("Best Individual: ", best)
    print("Best Fitness: ", best.fitness.values[0])

    print("Knapsack Items = ")
    knapsack.printItems(best)

    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()
