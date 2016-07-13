import random
from math import pow
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

#defining the make up of individuals and of the population
#bit = toolbox.attr_bool()
#ind = toolbox.individual()
#pop = toolbox.population(n=3)

#alpha = 1
sigma = 50.0 #alter based on broblem
#fitness sharing implimentation
def fitnessSharing(ind,pop):
    penalty = 0.0
    for x in range(len(pop)):
        d = hammingDistance(ind,pop[x])
        #print ind, pop[x]
        #print "d = ",d
        if d < sigma:           
            penalty += sharingValue(d)
    #print "penalty = ",penalty
    return penalty    

#severit of fitness penalty
def sharingValue(d):
    if d == 0:
        return 1
    else:
        return 1 - (d / sigma)        
        
#given two binart lists(individuals) returns the hamming distance between them
def hammingDistance(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def averageHammingDistance(pop):
    results, count = 0,0
    for i in combinations(pop,2):
        results += (hammingDistance(i[0],i[1]))
        count+=1
    return results/count
    
def momOfInertia2(pop): #hammingDistance not m o i
    diversity=0
    for i in xrange(len(pop)-1):
        for j in xrange(i+1,len(pop)):
            diversity+=hammingDistance(pop[i],pop[j])
    return diversity
    
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
    
#peaks is an array containing a bit string representing position and an integer representing height (higher is better)
#peaks =[[0,0,0,0,1,1,0,0,0,0],5]
#peaks = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],100,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],90]
peaks=[[0]*100,100,[1]*100,100]
def resetPeaks():
    global peaks
    #peaks = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],100,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],90]
    #peaks=[[0]*100,100,[0]*70+[1]*30,80]
    peaks=[[0]*100,100,[1]*100,100]


def movePeak(peak, amount):
    return np.roll(peak, amount)

def bitFlip(peak,prob):
    
    for i in xrange(len(peak)):
        if random.random() < prob:
            peak[i] = type(peak[i])(not peak[i])
    
    return peak

def evalOneMax(individual):
    return sum(individual),

def evalOneMaxShare(individual,pop):
    share= fitnessSharing(individual,pop)
    if share == 0:
        return sum(individual),
    else:
        return (sum(individual) - share),
    
def evalTwoMax(individual):        
    if sum(individual) >= (len(individual)/2):
        return sum(individual),
    else:
        return (len(individual)-sum(individual)),


#twomax fitness function
def evalTwoMaxShare(individual,pop):
    share = fitnessSharing(individual,pop)
    if share == 0.0:
        if sum(individual) >= (stringLength/2):
            return sum(individual),
        else:
            return (stringLength-sum(individual)),
    else:
        if sum(individual) >= (stringLength/2):
            return (sum(individual) - share),
        else:
            return (stringLength-sum(individual) - share),

def selRemovalGenotype(individuals, k):
    #select k individuals where none are duplicates
    chosen=[]
    population = sorted(individuals, key=attrgetter("fitness")) #not reversed
    count = len(population) - 1 #start at end of list, highest fitness
    while len(chosen) < k:
        if count == 0:
            return chosen
        if population[count] not in chosen:
            chosen.append(population[count])
        count -= 1
    return chosen

def fitnessCustom(individual):
    best = -10 #length of string
    for x in range (0, len(peaks), 2): #only take even items from list (thats the bit string)
       height = peaks[x+1]
       d = hammingDistance(individual, peaks[x])
       if height - d > best:
            best = height - d
    return float(best),
#get fitness of the custom function (also impimenting fitness ahring)
def fitnessCustomShare(individual,pop):
    best = -10 #length of string
    sharing = fitnessSharing(individual,pop)
    for x in range (0, len(peaks), 2): #only take even items from list (thats the bit string)
           height = peaks[x+1]
           d = hammingDistance(individual, peaks[x])
           if height - d > best:
                best = height - d
    if sharing == 0.0: #implimenting fitness sharing with custom fitness function
        return float(best),  
    else:
        return float(best - sharing),
        

#function to alter the peaks using bitwise or operator given peak index and the updated peak
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
    
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    resetPeaks()
    div=[]    
    logbook = tools.Logbook()
    logbook2 = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    logbook2.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    #invalid_ind = [ind for ind in population if not ind.fitness.valid]
    hofPop=toolbox.map(fitnessCustom,population) #change depending on eval func
    for ind, fit in zip(population, hofPop):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
        
    record = stats.compile(population) if stats is not None else {}
    div.append(momOfInertia(population))
    logbook2.record(gen=0, nevals=len(population), **record)
        
    fitnesses = []
    for i in xrange(len(population)):
        fitnesses.append(toolbox.evaluate((population)[i],population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(population) if stats is not None else {}
    div.append(momOfInertia(population))
    #div.append(averageHammingDistance(population))
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print logbook.stream
    
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
        #poppy = sorted(population, key=attrgetter("fitness"),reverse=True)
        #print poppy[0], poppy[0].fitness.values
        if gen==150 or gen == 300:
            print("Best individual is: %s\nwith fitness: %s" % (halloffame[0], halloffame[0].fitness))
            halloffame.clear()
            #print population
            #clear every time peaks move as new fitness period
            global peaks
            #peaks[0]=bitFlip(peaks[0],0.1)
            #peaks[2]=bitFlip(peaks[2],0.1)
            print population
            if gen==150:
                peaks[1]=80
                peaks[3]=100
            if gen==300:
                peaks[1]=100
                peaks[3]=80
            #peaks[0] = bitFlip(peaks[0],0.2)
            #print peaks[0]
            #print("Best individual is: %s\nwith fitness: %s" % (halloffame[0], halloffame[0].fitness))   
            
            for x in xrange(len(population)):
                del population[x].fitness.values
            
            fitnesses = []#toolbox.map(toolbox.evaluate, population+offspring)
            for i in xrange(len(population)):
                fitnesses.append(toolbox.evaluate((population)[i],population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
        
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
                       
               
        fitnesses = []#toolbox.map(toolbox.evaluate, population+offspring)
        for i in xrange(len(population+offspring)):
            fitnesses.append(toolbox.evaluate((population+offspring)[i],population+offspring))
        for ind, fit in zip(population+offspring, fitnesses):
            ind.fitness.values = fit
    
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}

        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print logbook.stream
        
        
        # Update the hall of fame with the generated individuals fitness not dependant on sharing
        hofPop = deepcopy(population)
        fitnesses=toolbox.map(fitnessCustom,hofPop) #CHANGE EVAL FUNCTION HERE
        for ind, fit in zip(hofPop, fitnesses):
            ind.fitness.values = fit
        
        if halloffame is not None:
            try:
                a=halloffame[0].fitness.values
            except IndexError:
                a=0
            halloffame.update(hofPop)
            b=halloffame[0].fitness.values
            if b>a:
                leaps+=1
        #record stats and offline and diversity and accuracy      
        record = stats.compile(hofPop) if stats is not None else {}
        div.append(momOfInertia(hofPop))
        logbook2.record(gen=gen, nevals=len(hofPop), **record)
        print logbook2.stream
        offline.append(halloffame[0].fitness.values[0])
        offlinegen.append(gen)
        accuracy.append((record["max"]-mint)/(maxt-mint))
        
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
            print "average bog: ", sum(logbook2.select("max"))/len(logbook2.select("max"))
            #print offline
            
            off.write(str(sum(offline)/len(offline)))
            off.write(",")
            bog.write(str(sum(logbook2.select("max"))/len(logbook2.select("max"))))
            bog.write(",")
            diver.write(str(div))
            diver.write(",")
            mini.write(str(logbook2.select("min")))
            mini.write(",")        
            aver.write(str(logbook2.select("avg")))
            aver.write(",")
            maxi.write(str(logbook2.select("max")))  
            maxi.write(",")
            
            #accuracy
            plt.figure(2)
            plt.title("accuracy")
            plt.plot(offlinegen,accuracy)            
            plt.axis([0,ngen,0,1])
            plt.show()
            #diversity
            plt.figure(3)
            plt.title("Diversity")
            plt.plot(xrange(len(div)),div)
            plt.axis([0,len(div),0,1250])
            plt.show()
            #min av max
            plt.figure(4)
            plt.title("min, avg, max")
            plt.plot(xrange(ngen+1),logbook2.select("min"),label="min")
            plt.plot(xrange(ngen+1),logbook2.select("avg"),label="avg")
            plt.plot(xrange(ngen+1),logbook2.select("max"),label="max")
            plt.legend(loc=8)
            plt.axis([0,ngen,0,maxt])
            plt.show()
            #stability
            plt.figure(5)
            plt.title("Stability")
            stability=[]
            for i in xrange(len(accuracy)):
                stability.append(max(0,accuracy[i]-accuracy[i-1]))
            plt.plot(offlinegen,stability)            
            plt.axis([0,ngen,0,1])
            plt.show()
            
            off.close()
            bog.close()
            mini.close()
            aver.close()
            maxi.close()
            diver.close()

            
    return population, logbook   


#toolbox.register("evaluate", evalTwoMaxShare)
toolbox.register("evaluate", fitnessCustomShare)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.010)
toolbox.register("select", selTournament, tournsize=3)
#toolbox.register("select", selRemovalGenotype)

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
    for i in xrange(1):
        pop, log, hof = main()
        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print pop