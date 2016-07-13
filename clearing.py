import random
from deap import base, creator, tools, algorithms, benchmarks
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
import random

from operator import attrgetter

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

def selRandom(individuals, k):
    return [random.choice(individuals) for i in xrange(k)]

#selTournament
def selTournament(individuals, k, tournsize):
    chosen = []
    counter =0
    flag=0
    pop = sorted(individuals, key=attrgetter("fitness"))    
    while flag==0:
        if pop[counter].fitness.values[0]>0.0:
            flag=1
        else:
            counter +=1
    
    if counter<30: #remove fitness 0 individuals if possible
        #print "pass"
        pop=pop[counter:]
        #print len(pop)
        if len(pop)<53:
            return pop
        for i in xrange(k):
            aspirants = selRandom(pop, tournsize)
            #print aspirants
            chosen.append(max(aspirants, key=attrgetter("fitness")))
            pop.remove(max(aspirants, key=attrgetter("fitness")))
        #print chosen
        return chosen
    else:
        #print "fail"
        for i in xrange(k):
            aspirants = selRandom(individuals, tournsize)
            #print aspirants
            chosen.append(max(aspirants, key=attrgetter("fitness")))
            individuals.remove(max(aspirants, key=attrgetter("fitness")))
        return chosen        
            
    

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    div=[] #for measuring diversity
    resetPeaks()    
    
    logbook = tools.Logbook()
    logbook2 = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    logbook2.header = ['gen', 'nevals'] + (stats.fields if stats else [])    
    
    
    fitnesses = []
    hofPop=toolbox.map(toolbox.evaluateHof,population) #change depending on eval func
    for ind, fit in zip(population, hofPop):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
        
    record = stats.compile(population) if stats is not None else {}
    div.append(momOfInertia(population))
    logbook2.record(gen=0, nevals=len(population), **record)
    
    
    #Evaluate the individuals with an invalid fitness
    
    #fitnesses = []
    #for i in xrange(len(population)):
    #    fitnesses.append(toolbox.evaluate((population)[i],population))                                                      
    #for ind, fit in zip(population, fitnesses):
    #    ind.fitness.values = fit

    population=toolbox.evaluate(population)    
    
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)
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
        if gen == 150 or gen ==300:
            print("Best individual is: %s\nwith fitness: %s" % (halloffame[0], halloffame[0].fitness))
            #print population
            halloffame.clear()
            #bog=[]
            global peaks
            peaks[0]=bitFlip(peaks[0],0.1)
            peaks[2]=bitFlip(peaks[2],0.1)
            #if gen==150:
            #    peaks[1]=80
            #    peaks[3]=100
            #if gen==300:
            #    peaks[1]=100
            #    peaks[3]=80
            for x in xrange(len(population)):
                del population[x].fitness.values
            
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluateHof, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        #for ind, fit in zip(invalid_ind, fitnesses):
        #    ind.fitness.values = fit
        
        # Update the fitness of all individuals for clearing
        clearingFit=toolbox.map(toolbox.evaluateHof,population + offspring)
        for ind, fit in zip(population+offspring, clearingFit):
            ind.fitness.values = fit
                
        
        #all individuals given fitness for hof this is then used to clear   
        #fitnesses = []
        #for i in xrange(len(population+offspring)):
        #    fitnesses.append(toolbox.evaluate((population+offspring)[i],population+offspring))                                                      
        #for ind, fit in zip(population+offspring, fitnesses):
        #    ind.fitness.values = fit
            
        tempPop=population+offspring
        tempPop = toolbox.evaluate(tempPop)

        # Select the next generation population
        population[:] = toolbox.select(tempPop, mu)#change temppop here if it doesnt work
    
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        
        logbook.record(gen=gen, nevals=len(population + offspring), **record)
        if verbose:
            print logbook.stream
            
        # Update the hall of fame with the generated individuals fitness not dependant on sharing
        hofPop = deepcopy(population)
        fitnesses=toolbox.map(toolbox.evaluateHof,hofPop) #change depending on eval func
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
        offline.append(halloffame[0].fitness.values[0])
        offlinegen.append(gen)
        accuracy.append((record["max"]-mint)/(maxt-mint))
        
        
        #show graphs at final generation
        if gen==ngen:
            #open files for data
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

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

stringLength=100

#registeringtoolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=stringLength)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#peaks = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],100,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],90]
peaks=[[0]*100,100,[1]*100,100]
moveRate=100

def resetPeaks():
    global peaks
    #peaks = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],100,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],90]
    peaks=[[0]*100,100,[1]*100,90]
    
#get fitness of the custom function
def fitnessCustom(individual):
    best = 0 #length of string
    for x in range (0, len(peaks), 2): #only take even items from list (thats the bit string)
       height = peaks[x+1]
       #print height
       d = hammingDistance(individual, peaks[x])
       #print height,"   ",d
       if height - d > best:
           #print height 
           best = height - d
    return float(best),

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
    
    
def peakDist(peak,pop):
    dist=[]
    for i in xrange(len(pop)):
        dist.append(hammingDistance(peak,pop[i]))
    return dist
#IMPORTANT TO ALTER
clearingRadius = 50 #size of the niches
nicheCap = 25 #number of individuals that get non 0 fitness in each niche

#need to add contingency for generation 0 where fitness is 0

#need to change evaluate here as well
#def clearing(individual,pop):
    #if individual.fitness == ind.fitness:
    #    return evalOneMax(individual) #change depending on fitness function to be used
#    local = getLocalPop(individual,pop)
#    local = sorted(local, key=attrgetter("fitness"), reverse=True)
#    if len(local) < nicheCap:
#        return evalTwoMax(individual)
#    elif individual.fitness > local[nicheCap-1].fitness or individual.fitness == local[0].fitness: # if equal same individual
#        return evalTwoMax(individual)
#    else:
#        return 0,

def clearing(pop):
    niches=0
    values=[]
    pop = sorted(pop, key=attrgetter("fitness"), reverse=True)
    for i in xrange(len(pop)):
        #print (pop[i].fitness.values[0]>0)
        if pop[i].fitness.values[0] >0:
            niches +=1
            nbWinners=1
            for j in xrange(i+1,len(pop)):
                #print hammingDistance(pop[i],pop[j])<clearingRadius
                if pop[j].fitness.values[0] >0 and hammingDistance(pop[i],pop[j])<clearingRadius:
                    if nbWinners<nicheCap:
                        nbWinners+=1
                    else:
                        pop[j].fitness.values=0.0,
                        #print pop.fitness.values
      
    for k in xrange(len(pop)):
        
        values.append(pop[k].fitness.values)        
    #print values
    #print pop
                 
    #print "niches: ",niches
    return pop

#def clearing(pop):
#    niches=[]
    #print pop
    #print pop[0].fitness.values
#    while len(pop)!=0:
#        local = getLocalPop(pop[0],pop)
        #print local
#        listSubtraction(pop,local) #remove local from pop
#        niches.append(local)
        #print pop
    #print niches[0]
#    print len(niches)
#    for i in xrange(len(niches)):
#        if len(niches[i])<nicheCap:
#            pop.extend(niches[i])
#        else:
#            cleared = niches[i][nicheCap:]
            #print cleared
#            for j in xrange(len(cleared)):
#                cleared[j].fitness.values = 0.0,
                #print j
#            pop.extend(niches[i])
    
    #pop=creator.Individual(pop)
    #print pop[0].fitness.values
    #return pop

def getLocalPop(individual,pop):
    local=[]
    #print "pop" 
    for i in xrange(len(pop)):
        if hammingDistance(pop[i], individual) <=clearingRadius: #potentially set minimum hamming distance to remove duplicates
            local.append(pop[i])
    return local

def listSubtraction(a,b):
    for i in xrange(len(b)):
        for j in xrange(len(a)):
            if a[j] == b[i]:
                del a[j] 
                break
    return a    
    
toolbox.register("evaluate", clearing)
toolbox.register("evaluateHof",fitnessCustom)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
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
    for i in xrange(30):    
        pop, log, hof = main()
        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print pop