import pandas as pd
import ga_compd_generation
import ga_crossing_mutation
import time
population_size = 100
mutation_rate = 0.1
cross_over_rate = 0.1

# df = ga_compd_generation.fitness(ga_compd_generation.population(population_size)).sort_values('Fitness', ascending=False)

def new_generations(Gen, population_size):
    half = int((population_size * 0.5)+1)
    selected = Gen.iloc[:half,:]
    # print(selected)
    new = [selected, ga_compd_generation.fitness(ga_compd_generation.population(half))]
    new_generation_input = pd.concat(new)
    new_generation_input.reset_index(drop=True, inplace=True)
    # print(new_generation_input)
    new_gen = ga_crossing_mutation.evolve_crossing(new_generation_input, cross_over_rate, mutation_rate)
    new_gen.reset_index(drop=True, inplace=True)
    # print(new_gen)
    # mut_pop = new_gen.drop_duplicates()
    # return mut_pop.head(population_size)
    return new_gen

#print('original', df, 'new', new_generations(df))
means = []
maxs = []
def Genetic_Algorithm(generation_number, population_size): #(population_size):
    Generation1 = ga_compd_generation.fitness(ga_compd_generation.population(population_size)).sort_values('Fitness', ascending=False)
    #print('Generation 1 and Fitness', Generation1.iloc[0][0], Generation1, '\n' )
    mean1 = Generation1['Fitness'][:len(Generation1) // 2].mean()
    max1 = Generation1['Fitness'].max()
    #print('mean1: ', mean1, 'max1:' , max1)
    #Generation1.to_csv('output/results/ovary/pop_size_50/t2/pop_size_' + str(population_size) + '_Generation_'+ str(generation_number) + '.csv')
    Generation1.to_csv('results/GA/liver/pop_size_' + str(population_size) + '_Generation_1.csv')
    Generation2 = ga_crossing_mutation.evolve_crossing(Generation1, cross_over_rate, mutation_rate)
    #print('Generation 2 and Fitness ', Generation2.iloc[0][0], Generation2, '\n')
    mean2 = Generation2['Fitness'].mean()
    max2 = Generation2['Fitness'].max()
    mean2 = Generation2['Fitness'][:len(Generation2) // 2].mean()
    #print('mean1: ', mean2, 'max1:', max2)
    #Generation2.to_csv('output/results/pop_size_50/t2/pop_size_' + str(population_size)+ '_Generation_'+ str(generation_number) + '.csv')
    Generation2.to_csv('results/GA/liver/pop_size_' + str(population_size)+ '_Generation_2.csv')
    Generation_next = Generation2
    means = [ mean1, mean2]
    maxs = [max1, max2]
    #print('means', means, 'maxs', maxs)
    g = 3
    i =  Generation2.iloc[0][0]
    #while i <= 100 and g in range(generation_number+1):
    while g in range(generation_number + 1):
        Generation_next = new_generations(Generation_next, population_size)
        i = Generation_next.iloc[0][0]
        # mean = Generation_next['Fitness'].mean()
        max = Generation_next['Fitness'].max()
        mean = Generation_next['Fitness'][:len(Generation_next) // 2].mean()
        # max = Generation_next['Fitness'][:len(Generation_next) // 2].max()

        #print('generation_number:', g, 'fitness', i, '\n')
        #print(Generation_next)


        #Generation_next.to_csv('output/results/pop_size_50/t2/pop_size_' +str(population_size)+ '_Generation_' + str(g) +'.csv')
        Generation_next.to_csv('results/GA/liver/pop_size_' + str(population_size) + '_Generation_' + str(g) + '.csv')

        means.append(mean)
        maxs.append(max)

        g += 1

    #print(means)
    #print(maxs)
    genn = generation_number + 1
    gens = list(range(1,genn))
    summary = pd.DataFrame( list(zip( gens, means, maxs)), columns= ['generations','mean', 'max'] )
    print(summary)
    #summary.to_csv('output/results/pop_size_50/t2/summary_pop_size_' + str(population_size) + '.csv')
    summary.to_csv('results/GA/liver/summary_pop_size_' + str(population_size) +'_gen_' + str(generation_number)+'.csv')
    return Generation_next

def final_loop():
    pop_col = []
    time_all = []
    gen_col = []
    gen = 100
    while gen <=100:
        population_size = 80
        while population_size <= 100:
            st = time.time()
            Genetic_Algorithm(gen, population_size)
            gen_col.append(gen)
            escape_time = time.time() - st
            time_all.append(escape_time)
            pop_col.append(population_size)
            print('Escape time:', escape_time)
            population_size += 10
        gen +=10
        et = pd.DataFrame(list(zip(pop_col, gen_col, time_all)), columns=['population_size','Generation number', 'Time'])
        et.to_csv('results/GA/liver/Time_' + str(population_size-10) + '.csv')

final_loop()