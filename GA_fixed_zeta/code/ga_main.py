import pandas as pd
import ga_compd_generation_new #compound_modified
import crossing_mutation_new
# import decoder_material
import time

population_size  = 10
#generation_number = 100
mutation_rate = 0.1
cross_over_rate = 0.1

#input
#normal_cell = []
#cancer_cell = []


#import warnings
#warnings.filterwarnings("ignore")
df = ga_compd_generation_new.fitness(ga_compd_generation_new.population(population_size)).sort_values('Fitness', ascending=False)

def new_generations(Gen, population_size):
    half = int((population_size * 0.5)+1)
    selected = Gen.iloc[:half,:]
    # print(selected)
    new = [selected, ga_compd_generation_new.fitness(ga_compd_generation_new.population(half))]
    new_generation_input = pd.concat(new)
    new_generation_input.reset_index(drop=True, inplace=True)
    # print(new_generation_input)
    new_gen = crossing_mutation_new.evolve_crossing(new_generation_input, cross_over_rate, mutation_rate)
    new_gen.reset_index(drop=True, inplace=True)
    # print(new_gen)
    # mut_pop = new_gen.drop_duplicates()
    # return mut_pop.head(population_size)
    return new_gen

#print('original', df, 'new', new_generations(df))
means = []
maxs = []
def Genetic_Algorithm(generation_number, population_size): #(population_size):
    Generation1 = ga_compd_generation_new.fitness(ga_compd_generation_new.population(population_size)).sort_values('Fitness', ascending=False)
    #print('Generation 1 and Fitness', Generation1.iloc[0][0], Generation1, '\n' )
    mean1 = Generation1['Fitness'].mean()
    max1 = Generation1['Fitness'].max()
    #print('mean1: ', mean1, 'max1:' , max1)
    #Generation1.to_csv('output/results/ovary/pop_size_50/t2/pop_size_' + str(population_size) + '_Generation_'+ str(generation_number) + '.csv')
    Generation1.to_csv('output/results/skin/pop_size_' + str(population_size) + '_Generation_1.csv')
    Generation2 = crossing_mutation_new.evolve_crossing(Generation1, cross_over_rate, mutation_rate)
    #print('Generation 2 and Fitness ', Generation2.iloc[0][0], Generation2, '\n')
    mean2 = Generation2['Fitness'].mean()
    max2 = Generation2['Fitness'].max()
    #print('mean1: ', mean2, 'max1:', max2)
    #Generation2.to_csv('output/results/pop_size_50/t2/pop_size_' + str(population_size)+ '_Generation_'+ str(generation_number) + '.csv')
    Generation2.to_csv('output/results/skin/pop_size_' + str(population_size)+ '_Generation_2.csv')
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
        mean = Generation_next['Fitness'].mean()
        max = Generation_next['Fitness'].max()

        #print('generation_number:', g, 'fitness', i, '\n')
        #print(Generation_next)


        #Generation_next.to_csv('output/results/pop_size_50/t2/pop_size_' +str(population_size)+ '_Generation_' + str(g) +'.csv')
        Generation_next.to_csv('output/results/skin/pop_size_' + str(population_size) + '_Generation_' + str(g) + '.csv')

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
    summary.to_csv('output/results/skin/summary_pop_size_' + str(population_size) +'_gen_' + str(generation_number)+'.csv')
    return Generation_next


#data_output = Genetic_Algorithm(population)
#data_output.to_csv('output/results/Fixed_Xgb_kidney_data_out.csv')
#final = Genetic_Algorithm(population_size=20)
#print(final.iloc[2])


#summary = pd.read_csv('output/results/summary.csv')
#plt.plot(summary['generations'], summary['mean'], color= 'red')
#plt.plot(summary['generations'], summary['max'], color= 'blue')
#plt.xlabel('Generations')
#plt.ylabel('Max/Mean Fitness')
#plt.show()



def final_loop():
    # population_size = 10
    pop_col = []
    time_all = []
    gen_col = []
    gen = 100
    while gen <=100:
        population_size = 10
        while population_size <= 100:
            st = time.time()
            Genetic_Algorithm(gen, population_size)
                #population_size += 10
            gen_col.append(gen)
                # gen += 10
            escape_time = time.time() - st
            time_all.append(escape_time)
            pop_col.append(population_size)
            print('Escape time:', escape_time)
            population_size += 10
        gen +=10
        et = pd.DataFrame(list(zip(pop_col, gen_col, time_all)), columns=['population_size','Generation number', 'Time'])
        #et.to_csv('output/results/pop_size_50/t2/Time_' + str(population_size-10) + '.csv')
        et.to_csv('output/results/skin/Time_' + str(population_size) + '.csv')

final_loop()


# def evaluation():
#     generation = [80, 90, 100]
#     for i in generation:
#         evaluate_generate_result = pd.read_csv('output/results/lungnew/pop_size_'+str(i)+'_Generation_100.csv')
#         evaluate = evaluate_generate_result.iloc[:, :-3]
#         evaluation = ga_compound_generation.result_evaluation(evaluate)
#         eval = pd.concat([evaluation, evaluate_generate_result.iloc[:,-3:]], axis=1)
#         eval.to_csv('output/results/lungnew/result_evaluation_'+str(i)+'.csv')
#
# evaluation()

