from math import ceil, floor
from random import randint, choice
from operator import itemgetter
import copy
import random

# Variables
code_secret: str = "JuniA2O23"
nb_population: int = 100
population: list = []
pct_elite: int = 0.2 #En pourcentage
taux_mutation: float = 0.1


def get_char() -> str:
    r = choice([(48,57),(65,90),(97,122)])
    return chr(randint(*r))


def create_population(population):
    for _ in range(nb_population):
        individu: list = []
        for _ in range(9):
            individu.append(get_char())
        population.append(copy.copy({"individu": individu}))


def calculate_score(population):
    for individu in population:
        score = 0
        for position_idx in range(len(code_secret)):
            if code_secret[position_idx] == individu["individu"][position_idx]:
                score += 1
        individu["score"] = score


def sort_by_score(population):
    return sorted(population, key=itemgetter('score'), reverse=True)


def selection_elite(population):
    nbr_elites = floor(len(population)*pct_elite)
    population_elite = population[:nbr_elites]
    population_nul = population[nbr_elites:]

    new_population = population_elite
    for nul in population_nul:
        if random.random() < 0.05:
            new_population.append(nul)

    return new_population


def reproduction(population):
    nb_enfant_generate: int = int(nb_population - len(population))
    population_enfant: list = []
    for _ in range(nb_enfant_generate):
        population_enfant.append(random.choice(population)["individu"][:4] + random.choice(population)["individu"][4:])

    for enfant in population_enfant:
        if random.random() < 0.1:
            enfant[random.randint(0,len(enfant)-1)] = get_char()
            {"individu": enfant}

    for enfant in population_enfant:
        population.append({"individu": enfant})
    return population


def main():
    global population
    create_population(population)
    while("".join(population[0]["individu"]) != code_secret):
        calculate_score(population)
        population = sort_by_score(population)
        population = selection_elite(population)
        population = reproduction(population)
    
    print("".join(population[0]["individu"]))


if __name__ == "__main__":
    main()
