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
peaks=[[0]*100,100,[1]*100,100]
moveRate=100

#registeringtoolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=stringLength)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def resetPeaks():
    global peaks
    peaks=[[0]*100,100,[1]*100,100]

def evalOneMax(individual):
    return sum(individual),

def evalTwoMax(individual):        
    if sum(individual) >= (len(individual)/2):
        return sum(individual),
    else:
        return (len(individual)-sum(individual)),

def fitnessCustom(individual):
    best = 0
    for x in range (0, len(peaks), 2): #only take even items from list (thats the bit string)
       height = peaks[x+1]
       #print height
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
    
def movePeak(peak, amount):
    return np.roll(peak, amount)
    
def bitFlip(peak,prob):
    for i in xrange(len(peak)):
        if random.random() < prob:
            peak[i] = type(peak[i])(not peak[i])
    
    return peak
    
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

    offline=[]
    offlinegen=[]
    accuracy=[]
    #leaps=0
    #needed for accuracy
    maxt=100 #highest peak
    mint=0
    # Begin the generational process
    for gen in range(1, ngen+1):
        #print peaks
        if gen==150 or gen ==300:
            halloffame.clear()
            global peaks 
            #peaks[0] = bitFlip(peaks[0],0.1)
            #peaks[2] = bitFlip(peaks[2],0.1)
            print population
            if gen==150:
                peaks[1]=80
                peaks[3]=100
            if gen==300:
                peaks[1]=100
                peaks[3]=80
            #print peaks
           
            for x in xrange(len(population)):
                 del population[x].fitness.values
            
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
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        
        div.append(momOfInertia(population))
        offline.append(halloffame[0].fitness.values[0])
        offlinegen.append(gen)
        accuracy.append((record["max"]-mint)/(maxt-mint))
        
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

        #show graphs at final generation
        if gen==ngen:
            #print leaps
            #off=open("offline.txt","a")
            #bog=open("bog.txt","a")
            #mini=open("min.txt","a")
            #aver=open("aver.txt","a")
            #maxi=open("max.txt","a")
            #diver=open("div.txt","a")
            print "offline performance: ",sum(offline)/len(offline)
            #off.write(str(sum(offline)/len(offline)))
            #off.write(",")
            #off.write("\n")
            print "average bog: ", sum(logbook.select("max"))/len(logbook.select("max"))
            #bog.write(str(sum(logbook.select("max"))/len(logbook.select("max"))))
            #bog.write(",")
            #bog.write("\n")
            #accuracy
            plt.figure(2)
            plt.title("accuracy")
            plt.plot(offlinegen,accuracy)            
            plt.axis([0,ngen,0,1])
            plt.show()
            #diversity
            #diver.write(str(div))
            #diver.write(",")
            plt.figure(3)
            plt.title("Diversity")
            plt.plot(xrange(len(div)),div)
            #plt.axis([0,len(div),0,maxt])
            plt.show()
            #min av max
            #mini.write(str(logbook.select("min")))
            #mini.write(",")       
            #mini.write("\n") 
            #aver.write(str(logbook.select("avg")))
            #aver.write(",")
            #aver.write("\n")
            #maxi.write(str(logbook.select("max")))  
            #maxi.write(",")
            #maxi.write("\n")
            #print logbook.select("min")
            #print " "
            #print logbook.select("avg")
            #print " "
            #print logbook.select("max")
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
            #off.close()
            #bog.close()
            #mini.close()
            #aver.close()
            #maxi.close()
            #diver.close()
        
    return population, logbook
    
#toolbox.register("evaluate", evalTwoMax)
toolbox.register("evaluate", fitnessCustom)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.010)
toolbox.register("select", tools.selTournament, tournsize=3)

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
    for i in xrange(10):
        pop, log, hof = main()
        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print pop