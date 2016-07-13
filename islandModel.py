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
    peaks=[[0]*100,100]#,[1]*100,100]

def movePeak(peak, amount):
    return np.roll(peak, amount)

def bitFlip(peak,prob):
    for i in xrange(len(peak)):
        if random.random() < prob:
            peak[i] = type(peak[i])(not peak[i])
    
    return peak
    
def evalOneMax(individual):
    return sum(individual),

def evalTwoMax(individual):        
    if sum(individual) >= (len(individual)/2):
        return sum(individual),
    else:
        return (len(individual)-sum(individual)),

#get fitness of the custom function
def fitnessCustom(individual):
    best = -10 #length of string
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
    
#toolbox.register("evaluate", fitnessCustom)
toolbox.register("evaluate", evalTwoMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", selTournament, tournsize=3)
toolbox.register("migrate", tools.migRing, k=1, selection=tools.selBest, replacement=random.sample)

def main():
    #random.seed(64)
    resetPeaks()
    div=[]  
    NBR_ISL = 3
    MU = 17 #adjusted so overall population is the same
    LAMBDA_ = 10
    NGEN = 450
    CXPB = 0.65
    MUTPB = 0.35
    MIG_RATE = 40   
    
    islands = [toolbox.population(n=MU) for _ in range(NBR_ISL)]
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "island", "evals", "std", "min", "avg", "max"
    
    for idx, island in enumerate(islands):
        for ind in island:
            ind.fitness.values = toolbox.evaluate(ind)    
        logbook.record(gen=0, island=idx, evals=len(island), **stats.compile(island))       
        hof.update(island)
    print(logbook.stream)
    
    div.append(momOfInertia(islands[0]+islands[1]+islands[2]))
    bog=[]    
    offline=[]
    offlinegen=[]
    accuracy=[]
    leaps=0
    #stability=[0]
    #needed for accuracy
    maxt=100 #highest peak
    mint=0
    gen = 1
    while gen <= NGEN:
        if gen ==1501 or gen ==3001:
            print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
            hof.clear()
            
            global peaks
            peaks[0]=bitFlip(peaks[0],0.10)
            #peaks[2]=bitFlip(peaks[2],0.10)
            #if gen==150:
            #    peaks[1]=80
            #    peaks[3]=100
            #if gen==300:
            #    peaks[1]=100
            #    peaks[3]=80
            for idx, island in enumerate(islands):
                #print island                
                for x in xrange(len(island)):
                    del island[x].fitness.values
            
                #invalid_ind = [ind for ind in island if not ind.fitness.valid]
                for ind in island:
                    ind.fitness.values = toolbox.evaluate(ind)
            
            
            
        for idx, island in enumerate(islands):
            
            offspring = algorithms.varOr(island, toolbox, LAMBDA_, cxpb=CXPB, mutpb=MUTPB)
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = toolbox.evaluate(ind)
            
            island[:] = toolbox.select(island + offspring, MU)
            
            logbook.record(gen=gen, island=idx, evals=len(offspring), **stats.compile(island))
            hof.update(island)
        print(logbook.stream)
        
        div.append(momOfInertia(islands[0]+islands[1]+islands[2]))
        offline.append(hof[0].fitness.values[0])
        offlinegen.append(gen)
        #accuracy.append((record["max"]-mint)/(maxt-mint))
        
            
        if gen % MIG_RATE == 0:
            toolbox.migrate(islands)
        gen += 1
        
        #show graphs at final generation
        if gen==NGEN+1:
            off=open("offline.txt","a")
            bog=open("bog.txt","a")
            mini=open("min.txt","a")
            aver=open("aver.txt","a")
            maxi=open("max.txt","a")
            diver=open("div.txt","a")
            #print div
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
            #plt.axis([0,NGEN,0,1])
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
            plt.plot((logbook.select("gen")),logbook.select("min"),label="min")
            plt.plot((logbook.select("gen")),logbook.select("avg"),label="avg")
            plt.plot((logbook.select("gen")),logbook.select("max"),label="max")
            #plt.legend(loc=8)
            plt.axis([0,NGEN,0,maxt])
            plt.show()
            #stability
            #plt.figure(5)
            #plt.title("Stability")
            #stability=[]
            #for i in xrange(len(accuracy)):
            #    stability.append(max(0,accuracy[i]-accuracy[i-1]))
            #plt.plot(offlinegen,stability)            
            #plt.axis([0,NGEN,0,1])
            #plt.show()
            off.close()
            bog.close()
            mini.close()
            aver.close()
            maxi.close()
            diver.close()
            
        
    
    return islands, logbook, hof

if __name__ == "__main__":
    random.seed(1)
    for i in xrange(3):
        pop, log, hof = main()
        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))  
    print pop
    