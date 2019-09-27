import random
import numpy
import math

#description
###############################################
'''
This is a simple model of reaction norm evolution along a bet-hedging-plasticity
continuum. Other models on bet-hedging/plasticity either assume two reaction norms
evolving (1 to a predictable cue = plasticity, 1 to white noise; e.g. Tufto 2015),
or an independent contribution of 1 reaction norm and an instability locus 
(e.g. Scheiner2014). This model assumes that there is a single polyphenic reaction 
norm evolving, its shape determines whether we call this strategy bet-hedging
or plasticity (see also  https://doi.org/10.1101/752881 and 10.32942/osf.io/trg34).
The model simulates a multivoltine species that has to diapause before winter onset,
but winter onset is not predictable across years. Depending on the standard 
deviation of winter onset, more plasticity or more bet-hedging is expected to evolve.

Each individual has four properties (b,c,d,e) that determine logistic reaction 
norm shape in response to t (time, e.g. day length). In each time step t, each 
individual decides whether to go to diapause, based on its reaction norm shape. 
If diapausing the individual is transferred to a seed bank (egg bank) of infinite size; 
if not, it will remain and reproduce. When t reaches winter onset,  all remaining 
individuals will be removed and the seed bank replaces the current population.
The next year begins and a new winter onset is drawn from a normal distribution 
with a given µ and sigma. 
A starting population has 100 individuals each from 10 genotypes 
(different bcde combinations), and the simulation runs for s = 10000 years. 
Growth rate is 3 offspring per individual and time step; µ =20, sigma = 0.

The situation is a bit different from standard models on plant germination, 
because the time to start diapausing varies, but all individuals hatch next year.
Still, the seed/egg bank ensures survival under bad and unpredictable conditions.

All variables are integers unless specified otherwise
'''
#definition of constants
###############################################
growth_rate = 1.2
popsize = 1000

mut_rate = 1/(popsize) #mutation rate

mu_float = 20 #mean winter onset
sigma_float = 0 #sd in winter onset
max_year = 500 #number of years to simulate

mean_rate = 0 #directional change of winter onset with time (climate change)
end = round(50 + max_year*mean_rate) #end of year, after which surival = 0
surv_ratio = 0.2 # at every time step after winter onset, 
#most of the remaining individuals die before reproducing
K = math.ceil(growth_rate**(mu_float + 2* sigma_float) * popsize) #carrying capacity
#should only be exceeded in 2% of all years

#functions
###############################################
def determine_diapause(parms, t): #f(t) = logit(t,b,c,d,e)
    #input: list with 4 reaction norm parameters, and t
    b,c,d,e = parms #slope,lower,upper,inflection point
    diap_probability = d/(1+math.exp(-b*(t-e))) 
    out_bool = bool(numpy.random.binomial(1,diap_probability))
    return (out_bool) #singlevalue


def reproduce(individ_list): 
    #input: list of individuals that reproduce
    off = []
    r = math.floor(growth_rate)
    fract = growth_rate - r
    if random.uniform(0,1) < fract:
        r +=1 
    for i in range(1,r):
            off.append(individ_list) #need to give new id
    return (off)

def mutate_offspring (off, mut_rate):
    #input: a list of individuals
    off_new = []
    for i in off:
        b,c,d,e = i[1:5]
        if random.uniform(0,1) < mut_rate:
            b = random.gauss(b,0.1)
            c = random.gauss(c,0.1)
            if c < 0: 
                c = 0
            if c > 1: 
                c =1
            d = random.gauss(c,0.1)
            if d < 0: 
                d = 0
            if d > 1: 
                d =1
            e = random.gauss(e, 1)
        off_new.append([i[0],b,c,d,e,False])
        #could decouple mutations to 4 parameters    
    return (off_new)
    
#initialization
###############################################
pop_list = [] #create population (list), in which each individual's reaction 
#norm shape and diapausing state is saved
for i in range(0,popsize):
    pop_list.append([i,0.5, 0, 1, mu_float,False])
    # pop_list.append([i,random.uniform(0,3), 0, 
     #                1, random.uniform(0,40),False])   
    #ID, b,c,d,e paramters of log curve (as in drc package of R), current diapause state
egg_list = mutate_offspring(pop_list, mut_rate)
#the starting population will hatch from these. egg_list generally holds all diapausing
#individuals, until they are released at t = 0 of the next year
print("estimated max. popsize: ", K)

#main program
###############################################
for year in range(0, max_year):
    print("year", year)#, len(egg_list), "eggs from seed bank") 
    if egg_list == []:
        print("population extinct after year ", year)
        break  
    if len(egg_list) > popsize:
        pop_list = random.sample(egg_list,popsize) #winter carrying capacity
        #this reduces population size to 1000 at the beginning of the year, the
        #population grows from here over the season (with r = 1.2, popsize = 1000
        #and mu = 20 its ~38,000 at the end of each year)
    else:
        pop_list = egg_list
    egg_list = []

    #calculate winter onset in this year:
    t_on = random.normalvariate(mu_float, sigma_float) + mean_rate*year
    t_on = round(t_on)
    print("winter onset: ", t_on)
    
    #go through each timestep of the year and reproduce/diapause:    
    for t in range(0, end):
        if t > t_on:
            pop_list = random.sample(pop_list, round(len(pop_list)*surv_ratio))
            #round also converts to int, so 99*0.2 is no problem
        if (pop_list == []): #may occur as t approaches t_on and 
            # means that all remaining individuals made it to the seed bank
            break #does not stop progam, jumps only to end of t-loop
            
        #density dependent regulation: consider removing
        #if len(pop_list) > K:
        #    pop_list = random.sample(pop_list,K) 
            
        #determine diapause of each individual:
        dstate = []
        nl = []
        x=0
        for ind in pop_list: #ind = individual
            dip = determine_diapause(ind[1:5], t)
            newline = [x, ind[1], ind[2], ind[3], ind[4], dip] #give new id
            dstate.append(dip) #report for debugging
            nl.append(newline)
            x+=1
        pop_list = nl     
     #   print("popsize: ", len(pop_list), "diapause fraction:", sum(dstate)/len(dstate))
        
        #determine new offspring and remove diapausers to seed bank:
        offspring = []
        rem_index = []
        
        for ind in pop_list:
            if ind[5] == False: #not diapausing
                offspring.extend(reproduce(ind))
            else:
                rem_index.append(ind[0])       
        
        #update seed bank and remove from pop
        new_eggs = []
        for index in sorted(rem_index, reverse= True):
            candidate = pop_list[index]
            new_eggs.append(candidate)
            del pop_list[index] #delete the pop_list entry with matchin ID
        new_eggs = mutate_offspring(new_eggs, mut_rate)
        offspring = mutate_offspring(offspring, mut_rate) 
        egg_list.extend(new_eggs)
        pop_list.extend(offspring)
        
  #      print("day ", t, "N ", len(pop_list), "egglist: ", len(egg_list))
    
        
    means =[]
    slopes = []
    lower = []
    upper = []
    for i in range(1, len(egg_list)):
        means.append(egg_list[i][4])
        slopes.append(egg_list[i][1])
        lower.append(egg_list[i][2])
        upper.append(egg_list[i][3])
        
        
    print("e: ", round(sum(means)/len(means),2), "b: ", round(sum(slopes)/len(slopes),2),
          "c: ", round(sum(lower)/len(lower),2), "d: ", round(sum(upper)/len(upper),2))


#count results
print("finished.")


