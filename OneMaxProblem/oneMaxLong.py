from deap import base
from deap import tools
from deap import creator

import random

import matplotlib.pyplot as plt
import seaborn as sns

IND_LENGTH = 200

POPULATION_SIZE = 300
MAX_GENERATIONS = 70

P_MUTATION = 0.2
P_CROSSOVER = 0.9

toolbox = base.Toolbox()

# function to create an individual
toolbox.register("zeroOrOne", random.randint, 0, 1)

# fitness definition
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# individual definition
creator.create("Individual", list, fitness=creator.FitnessMax)

# register the function to create an individual
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, IND_LENGTH)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def oneMaxFitness(Individual):
    return sum(Individual),


# register the function to calculate the fitness value
toolbox.register("evaluate", oneMaxFitness)

# registering the genetic operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / IND_LENGTH)


# Algorithm's flow
def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationsCounter = 0

    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    fitnessValues = [individual.fitness.values[0] for individual in population]

    maxFitnessValues = []
    meanFitnessValues = []

    while max(fitnessValues) < IND_LENGTH and generationsCounter < MAX_GENERATIONS:
        generationsCounter += 1
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring
        fitnessValues = [ind.fitness.values[0] for ind in population]
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)

        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}".format(generationsCounter, maxFitness, meanFitness))
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index], "\n")

    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()
