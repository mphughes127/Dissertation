import random
from deap import base, creator, tools, algorithms, benchmarks
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
import random

from operator import attrgetter

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

stringLength=100

#registeringtoolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=stringLength)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

peaks=[[0]*100,100,[1]*100,100]
def resetPeaks():
    global peaks
    peaks=[[0]*100,100,[1]*100,100]
    
def movePeak(peak, amount):
    return np.roll(peak, amount)

def selRandom(individuals, k):
    return [random.choice(individuals) for i in xrange(k)]

#selTournament
def selTournament(individuals, k, tournsize):
    chosen = []
    for i in xrange(k):
        aspirants = selRandom(individuals, tournsize)
        #print aspirants
        chosen.append(max(aspirants, key=attrgetter("fitness")))
        individuals.remove(max(aspirants, key=attrgetter("fitness")))
    return chosen
    
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    resetPeaks()
    div=[]    
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    div.append(momOfInertia(population))
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    bog=[]    
    offline=[]
    offlinegen=[]
    accuracy=[]
    leaps=0
    #stability=[0]
    #needed for accuracy
    maxt=100 #highest peak
    mint=0
    # Begin the generational process
    for gen in range(1, ngen+1):
        #move the peaks
        if gen==150 or gen == 300:
            print("Best individual is: %s\nwith fitness: %s" % (halloffame[0], halloffame[0].fitness))
            halloffame.clear()
            
            global peaks
            peaks[0]=bitFlip(peaks[0],0.10)
            peaks[2]=bitFlip(peaks[2],0.10)
            if gen==150:
                peaks[1]=80
                peaks[3]=100
            if gen==300:
                peaks[1]=100
                peaks[3]=80
            
            for x in xrange(len(population)):
                del population[x].fitness.values
            
            #invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
        
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            try:
                a=halloffame[0].fitness.values
            except IndexError:
                a=0
            halloffame.update(offspring)
            b=halloffame[0].fitness.values
            if b>a:
                leaps+=1
        
        
        
        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        offline.append(halloffame[0].fitness.values[0])
        offlinegen.append(gen)
        accuracy.append((record["max"]-mint)/(maxt-mint))
        div.append(momOfInertia(population))
        accuracy.append((record["max"]-mint)/(maxt-mint))
        #if gen!= 0 and gen!= 1:
            #print accuracy[gen-1]
            #print accuracy[gen-2]
            #stability.append(max(0,accuracy[gen-1]-accuracy[gen-2]))
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream
                
        
        #show graphs at final generation
        if gen==ngen:
            off=open("offline.txt","a")
            bog=open("bog.txt","a")
            mini=open("min.txt","a")
            aver=open("aver.txt","a")
            maxi=open("max.txt","a")
            diver=open("div.txt","a")
            
            print leaps
            print "offline performance: ",sum(offline)/len(offline)
            print "average bog: ", sum(logbook.select("max"))/len(logbook.select("max"))
            #print offline
            off.write(str(sum(offline)/len(offline)))
            off.write(",")
            bog.write(str(sum(logbook.select("max"))/len(logbook.select("max"))))
            bog.write(",")
            diver.write(str(div))
            diver.write(",")
            mini.write(str(logbook.select("min")))
            mini.write(",")        
            aver.write(str(logbook.select("avg")))
            aver.write(",")
            maxi.write(str(logbook.select("max")))  
            maxi.write(",")
            
            #accuracy
            #plt.figure(2)
            #plt.title("accuracy")
            #plt.plot(offlinegen,accuracy)            
            #plt.axis([0,ngen,0,1])
            #plt.show()
            #diversity
            plt.figure(3)
            plt.title("Diversity")
            plt.plot(xrange(len(div)),div)
            plt.axis([0,len(div),0,1250])
            plt.show()
            #min av max
            plt.figure(4)
            plt.title("min, avg, max")
            plt.plot(xrange(ngen+1),logbook.select("min"),label="min")
            plt.plot(xrange(ngen+1),logbook.select("avg"),label="avg")
            plt.plot(xrange(ngen+1),logbook.select("max"),label="max")
            plt.legend(loc=8)
            plt.axis([0,ngen,0,maxt])
            plt.show()
            #stability
            #plt.figure(5)
            #plt.title("Stability")
            #stability=[]
            #for i in xrange(len(accuracy)):
            #    stability.append(max(0,accuracy[i]-accuracy[i-1]))
            #plt.plot(offlinegen,stability)            
            #plt.axis([0,ngen,0,1])
            #plt.show()
            off.close()
            bog.close()
            mini.close()
            aver.close()
            maxi.close()
            diver.close()
            
    return population, logbook  

def evalOneMax(individual):
    return sum(individual),

def evalTwoMax(individual):        
    if sum(individual) >= (len(individual)/2):
        return sum(individual),
    else:
        return (len(individual)-sum(individual)),

#get fitness of the custom function
def fitnessCustom(individual):
    best = 0 #length of string
    for x in range (0, len(peaks), 2): #only take even items from list (thats the bit string)
       height = peaks[x+1]
       d = hammingDistance(individual, peaks[x])
       if height - d > best:
            best = height - d
    return float(best),

#given two binart lists(individuals) returns the hamming distance between them
def hammingDistance(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def averageHammingDistance(pop):
    results, count = 0,0
    for i in combinations(pop,2):
        results += (hammingDistance(i[0],i[1]))
        count+=1
    return results/count
    
def momOfInertia(pop):
    diversity=0
    centroid=getCentroid(pop)
    for i in xrange(stringLength):
        for j in xrange(len(pop)):
            diversity+=pow(pop[j][i]-centroid[i],2)
    return diversity
    
def getCentroid(pop):
    centroid=[0.0]*stringLength
    for i in xrange(stringLength):
        for j in xrange(len(pop)):
            centroid[i]+=pop[j][i]
            
    return [x/len(pop) for x in centroid]

def alterPeaks(peak, update):
    for x in range(10):
        peaks[peak][x] = peaks[peak][x] ^ update[x] #bitwise or operator to update peak bit string
    peaks[peak + 1] = update[10] #update height length currently hardcoded
    print peaks

def bitFlip(peak,prob):
    for i in xrange(len(peak)):
        if random.random() < prob:
            peak[i] = type(peak[i])(not peak[i])
    
    return peak
    
def selRemovalGenotype(individuals, k):
    #select k individuals where none are duplicates
    chosen=[]
    population = sorted(individuals, key=attrgetter("fitness")) #not reversed
    count = len(population) - 1 #start at end of list, highest fitness
    while len(chosen) < k:
        if count == 0:
            print len(chosen)
            return chosen
        if population[count] not in chosen:
            chosen.append(population[count])
        count -= 1
    return chosen
    
#toolbox.register("evaluate", evalTwoMax)    
toolbox.register("evaluate", fitnessCustom)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.010)
toolbox.register("select", selRemovalGenotype)

def main():
    import numpy
    
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values) #values
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)


    pop, logbook = eaMuPlusLambda(pop, toolbox, 50, 30, cxpb=0.65, mutpb=0.35, ngen=450, stats=stats, halloffame=hof, verbose=True)
    
    return pop, logbook, hof
    
if __name__ == "__main__":
    random.seed(1)
    for i in xrange(30):
        pop, log, hof = main()
        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print pop