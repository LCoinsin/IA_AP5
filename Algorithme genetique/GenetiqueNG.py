from math import ceil, floor
from random import randint, choice, random
from operator import itemgetter
import copy


# Variables
code_secret: str = "JuniA2O23"
population_taille: int = 100
population: list = []
pct_elite: int = 0.2 #En pourcentage
taux_mutation: float = 0.1


def random_char() -> str:
    r = choice([(48,57),(65,90),(97,122)])
    return chr(randint(*r))


class individu:
    genetique: list = []
    score: int = 0

    def __init__(self):
        self.genetique = self.__set_genome()
        self.__set_score() # Calcul du score

    def __set_genome(self) -> list:
        return [random_char() for _ in range(len(code_secret))]

    def set_genome_from_parent(self, a, b) -> None:
        self.genetique = a + b

    def __set_score(self) -> int:
        for idx in range(len(code_secret)):
            if code_secret[idx] == self.genetique[idx]:
                self.score += 1

    def set_mutation(self) -> None:
        self.genetique[randint(0, len(self.genetique)-1)] = random_char()
        self.__set_score()


def create_population():
    return [individu() for _ in range(population_taille)]


def sort_population_by_score(population):
    population.sort(key=lambda x : x.score, reverse=True)


def selection_elite(population) -> list:
    nbr_elites = floor(len(population)*pct_elite)
    population_temp: list = population[:nbr_elites]

    for individu_nul in population[nbr_elites:]:
        if random() < 0.05:
            population_temp.append(individu_nul)

    return population_temp


def reproduction(population):
    population_enfants: list = list()
    for _ in range(int(population_taille - len(population))):
        enfant_tmp = individu()
        enfant_tmp.set_genome_from_parent(
            choice(population).genetique[:4],
            choice(population).genetique[4:]
        )
        if random() < taux_mutation:
            enfant_tmp.set_mutation()
        population_enfants.append(enfant_tmp)

    for enfant in population_enfants:
        population.append(enfant)

    return population


def main():
    total = 0
    population = create_population()
    while("".join(population[0].genetique) != code_secret):
        print(total , " - ", "".join(population[0].genetique))
        sort_population_by_score(population=population)
        population = selection_elite(population=population)
        population = reproduction(population=population)
        total += 1
    print("".join(population[0].genetique))


if __name__ == "__main__":
    main()
