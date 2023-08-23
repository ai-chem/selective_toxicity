import random
import ga_compd_generation_new
in1 = ['HepG2', 'AlamarBlue', 'TiO2', 'original', 'Human', 'Human', 'epithelial', 'liver', 'Carcinoma', 48, 600.0, 549.0, 20.02, 5.369127517, 2.806666667, -0.93, 1.4, 16, 79.865, 2, 0, 0, -0.2401, 2.878049394, 1.683250823, 0.0, 0.314285714, 3.314285714, 76.53795327071668, 74.7304359843506, 1.8075172863660782]
in2 = ['HepG2', 'Alamar', 'C', 'original1', 'Human1', 'Human1', 'epithelial1', 'liver1', 'Carcinoma1', 12, 19.0, 39.7, 8.29, 0.0, 2.55, 0.0, 0.7, 4, 12.011, 0, 0, 0, 0.08129, 0.5, 0.0, 0.0, 0.0, 0.0, 73.8215991847434, 72.4381269946488, 1.3834721900946079]
indv2_list = ga_compd_generation_new.fitness(ga_compd_generation_new.population(size=50))
# print(indv2_list.loc[2].values.tolist())
cross_over_frequency = 0.2
mutation_rate = 0.2

def to_crossover(indv1, indv2, cross_over_frequency):
    a = random.random()

    for each in range(1,len(indv1)):
        if (each ==1) or (each ==9) or (each == 10) or (each == 11) or (each == 12):
            if random.random()< cross_over_frequency:
                indv1[each] = indv2[each]
            continue
        if a < cross_over_frequency:
            indv1[each] = indv2[each]

    return indv1

# print((to_crossover(in1, in2, cross_over_frequency=0.5)),'\n')

def to_mutation(individual1, mutation_rate):
    individual2 = indv2_list.iloc[random.randrange(20)].values.tolist()
    # print(individual2)
    mut = to_crossover(individual1, individual2, mutation_rate)
    return mut
# print(len(in1),to_mutation(in1, mutation_rate=0.1))

# print(to_mutation(in1, mutation_rate))