# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

'''
This model explores the costs and limits of reaction norm evolution (costs and limits
of phenotypic plasticity, bet-hedging and canalization). 
The scenario is the same as in Joschinski & Bonte, Frontiers in Ecology and Evolution 2020.

In short, we consider a haploid genotype that faces the decision to produce one 
of two phenotypes (P1 and P2) under uncertain environmental conditions. 
An environment E can take two discrete states (E1 and E2), and the change in environmental 
state (such as onset of winter) is imperfectly correlated to a cue c:
The change in E is normally distributed and the mean timing of environmental change
corresponds to c = 5 with a standard deviation sigma; thus the cumulative probability 
function of the normal distribution describes the probability of E2 given a cue c
(e.g. with sigma = 1: the probability of E2 is 0.5 when is c = 5, 0.84 for c = 6, and 0.999 for c = 8).
The genotype determines a reaction norm, which describes the proportion of P2 
in response to c, without knowing E directly.

A starting population has 3000 individual genotypes that vary in reaction 
norm shapes to c (c ranges from 0 to 10 in arbitrary units). In each time step 
one environmental cue is chosen and an according environment is created. For example, 
a cue c = 6 may be chosen, which may correspond to a probability of E2 = 0.84 
(depending on sigma); the environment is hence likely to be E2 in this year. 
The same cue c is presented to all genotypes, which then choose phenotypes for their 
offspring according to their reaction norm shape. We initially assume that each 
genotype has 100 individual offspring that can react to c. For example, the genotype's 
reaction norm may determine that 60% of the offspring shall be of type P2 at c = 6. 
Around 60 of the offspring will thus be P2, and 40 will be P1 (because these are 
probabilities, the exact number of offspring varies).
All phenotypes are then subjected to the same E (E2 in the example) and the fitness
is calculated. In the standard scenario the fitness of P1 is 4 in E1, but zero in E2, 
while the fitness of P2 is 1 regardless of environmental conditions. The genotype
fitness is calculated as the arithmetic mean fitness of the offspring.
In the example the environment was E2, so about 40 offspring have 0 fitness (P1 in E2), while
60 offspring have 1 fitness (P2 in E2), i.e. the average fitness is 0.6. In another year
with c = 6 the environment may turn out E1 (with 16% probability), and the fitness would
be (40 * 4 + 60 * 1)/100 = 2.2

Before the next year starts the genotypes reproduce with a growth rate that equals
their fitness. The reaction norms may mutate with a probability of 1/1000 in the process. 
In the next time step the reproduced genotypes replace the current population. If the
available genotypes exceed the population size (3000), a random sample equalling the 
populatoin size is drawn from the pool of available genotypes. 

The model runs for 2000 time steps, after which population size and reaction norm 
 parameters (Var_among, var_within, mean) are recorded.
 
 We will change the fitness functions in various runs and additionally impose 
 constraints on the reaction norm shape as well as on the environmental conditions.


'''

import random
import numpy
import math
import matplotlib.pyplot as plt
from scipy import stats
import copy
import os

'''classes'''
class Genotype(object):
    ''' Creates an individuum with a reaction norm shape determined by 10 parameters'''
    
    def __init__(self,p_list = [], costs = [], costmag = []):
        self.p_list = p_list if p_list else [
                numpy.random.uniform(0,1) for i in range(10)]
        #probability of phenotype P2 as function of a cue c
    
        
    def __str__(self):
        return(str(self.p_list)) 
    
    def reproduce(self, r, costs = [], costmag = 0): 
        '''reproduces the genotype r times
        
        r = 4.2 will return a list with 4 individuals in 80% of all cases, and with 5
        individuals in the remaining 20% of all cases. expected input: r = float > 0'''
        if costs == "s": #for the moment these are only phenotypic variance costs
            t = self.pars()
            c = (t[1]+t[2])/0.25 #makes phenotypic variance go from 0-1    
        elif costs == "r":
            t = self.pars()
            c = t[1]/(t[1]+t[2]) #variance composition, goes from 0-1
        else:
            c = 0
            
        r = r + r * costmag * c 
        n = math.floor(r) + numpy.random.binomial(1, r- math.floor(r),1)[0]
        return( [copy.deepcopy(self) for i in range(0,n)] )  
             
    def mutate(self, mut_frac):
       '''induces mutations in the genotype's reaction norm'''
         
       self.p_list = [i if (random.uniform(0,1) > mut_frac) else 
                      random.uniform(0,1) for i in self.p_list]

    def phenotype (self, c):
        return(bool(numpy.random.binomial(1,self.p_list[c],1)[0]))
        #is phenotype of form P2?
        
    def pars(self):
        ''' calculate variance among and within environments

        var_within = sum p*(1-p) / n
        var_among = sd^2'''
        #these terms are used as a summary statistic of the reaction norm shape
        #a description can be found at 
        #https://doi.org/10.3389/fevo.2020.517183
        p = self.p_list
        p2 =[i * (1-i) for i in p]
        f = sum(p)/10
        among = numpy.std(p)**2
        within = numpy.mean(p2) 
        return([f, among, within]) 
    

        
        
class Run_Program(object):
    '''run the model'''
    def __init__ (self, model_name = "generic", max_year = 2000, saving =False, 
                  env = [5,3], startpop = [],
                  popsize = 3000, mut_rate = 1/1000,
                  replicates = 100,
                  gr = numpy.array([[4,1],[0,1]]),
                  costs = [], costmag = 0, costtype = "maintenance"):

        '''saves parameters and creates starting population'''
        
        self.model_name = model_name
        self.max_year = max_year #number of time steps
        self.saving = saving #boolean: should details of reaction norms be saved at 
        #each time step? Set to false, except for debugging purposes
        self.env = env #mean and variance of the environment
        self.startpop = startpop
        self.popsize = popsize #population size at start of the year
        self.mut_rate = mut_rate
        self.replicates = replicates #replicates of the genotype
        
        self.costs = costs #for the moment only phenotypic variance costs; later costs should be
        #[], "s","r" or similar. negative costs are costs of phenotypic variance, positive
        #values are benefits of variance/costs of canalization
        self.costmag = costmag #can later be used for the magnitude of any cost now
        #only phenotypic variance costs
        self.cost_type = costtype #"p1" will incur these costs only on phenotype P1, 
            #"p2" on phenotype P2, "e1" only in environment 1 and "e2" only in E2; 
            #"maintenance", "m" or any other term will incur the costs in any environment
            #and independent of the phenotype
        
        self.gr = gr- gr * 0.5*self.costmag #growth rates of the 2 phenotypes in the 2 environemnts
        #sorting: [[E1 P1, E1 P2], [E2 P1, E2 P2]]
        #e.g. insect diapause: [[summer, nondiapausing = 4; summer diapasuing = 1],
        #[winter nondiapausing = 0, winter diapausing  = 1]]
        #for an average reaction norm (with e.g. intermediate phenotypic variance) 
        #a cost of costmag/2 applies. This is substracted from growth rate here to 
        #standardize (costfree have on average same growth rates)
        
        self.fitness_list = [] #stores no. survivors per time step
        self.details_list = [] #stores reaction norm shape at each time step
        self.sample_list = [] #stores a sample of 100 genotypes in each time step
        self.c_list =[] #stores the environmental cue that was drawn
        self.e_list = [] #stores the environment that was drawn
        self.eggs = startpop
        if not self.eggs:
            self.eggs = [Genotype() for i in range(self.popsize)]
                
        print("Initialized model '", model_name, "'")
        
    def __str__(self):
        first_line = str("model: {}\n".format(self.model_name))
        second_line = str("Years: {:6}   N:{:4}   mutrate: {:4.6f}\n".format(
                self.max_year, self.popsize, self.mut_rate))
        third_line = str("r: {:5.2f},{:5.2f},{:5.2f},{:5.2f}\n".format(
                self.gr[0,0], self.gr[0,1], self.gr[1,0], self.gr[1,1]))
        fourth_line = str('individuals: {:2}'.format(
                self.replicates))
        return(first_line+ second_line+ third_line+ fourth_line)
        
        
    def run(self):
        '''wrapping method that runs the model, handles extinction, and saves output'''
        print("\nmodel: ", self.model_name, "\nRunning. 1 dot = 10 years")
       
        yc = 1 #saves year of extinction, if applicable
        extinct_at = []
          
        for i in range(0, self.max_year): 
            if self.eggs:
                l = min(len(self.eggs), self.popsize)
                alive_list = random.sample(self.eggs, l)   
                for genotype in alive_list:
                    genotype.mutate(self.mut_rate)
                self.eggs = self.runyear(alive_list)        
                self.fitness_list.append(len(self.eggs))
                if self.saving: #mostly for debugging only
                    self.sample_list.append(
                            random.sample(self.eggs,min(len(self.eggs),100)))
                    self.details_list.append(self.save_rn(self.eggs))
                yc +=1
                if not yc % 10: 
                    print(".", end = '')
            elif extinct_at:
                pass
            else:
                extinct_at = yc
        if extinct_at:
            print ("Population extinct in year", extinct_at)
        print ("run complete. Saving.")
        self.results = self.eggs
        if self.saving:
            self.details_list = numpy.array(self.details_list)
        return(yc)
        
    def runyear(self, curr_list):
        '''In every year an environment and according cue are created. Genotype survival
        is tested against this environment
        
        To reduce stochasticity due to Bernoulli variance, each genotype is replicated
        100 times and the average fitness of these 100 genotypes (all receiving 
        same environment and cue) is presented'''

        c = random.choice(range(10))
        self.c_list.append(c)
        e = stats.norm.cdf(c, self.env[0], self.env[1])
        E = numpy.random.binomial(1,e,1)[0] #0 in summer(E1), 1 in winter(E2)   
        #these paramters are the same for every replicate and for every genotype!
        self.e_list.append(E)
        offspring_list = []
        for genotype in curr_list:
            replicate_fitness =[]
            for r in range(self.replicates):
                phen = int(genotype.phenotype(c)) #this is 0 when phenotype P1 is
                                                #chosen, and 1 for P2
                f = self.gr[E,phen] 
                if self.cost_type == "p1": 
                    replicate_fitness.append(f + f * 2 * self.costmag * (1-int(phen))) 
                elif self.cost_type == "p2":
                    replicate_fitness.append(f + f * 2 * self.costmag * int(phen)) 
                else: 
                    replicate_fitness.append(f + f * self.costmag)
                
                
            if self.cost_type =="e1":
                offspring_list.extend(genotype.reproduce(numpy.mean(replicate_fitness), 
                                                     self.costs, 2 * self.costmag*(1-E)))
            elif self.cost_type =="e2":
                offspring_list.extend(genotype.reproduce(numpy.mean(replicate_fitness), 
                                                     self.costs, 2 * self.costmag*E))
            else:
                offspring_list.extend(genotype.reproduce(numpy.mean(replicate_fitness), 
                                                     self.costs, self.costmag))
                
        return(offspring_list)

    
    def save_rn (self, eggs): #only used for debugging, i.e. when saving = True
        '''saves reaction norm parameters (var_within, var_among, ratio, sum, midpoint)'''
        x = [individual.pars() for individual in eggs]
        f =[]
        among=[]
        within=[]
        ratio = []
        varsum =[]
        for i in x:
            f.append(i[0])
            am = round(i[1],2)
            wi = round(i[2],2)
            among.append(am)
            within.append(wi)
            ratio.append(am/(am+wi) )
            varsum.append(am+wi)
        return([f, among, within, ratio, varsum])
   

    def save_data(self): #the actual function for saving all important results
        os.mkdir(self.model_name)
        numpy.save(os.path.join(os.getcwd(), self.model_name, "popsize"),
                       self.fitness_list) #model name twice to make generic.npy in subfolder "generic"
        numpy.save(os.path.join(os.getcwd(), self.model_name, "startpop"), 
                       self.startpop)
        numpy.save(os.path.join(os.getcwd(), self.model_name, "results"), 
                      numpy.array([self.results[i].p_list for i in range(len(self.results))]))
        numpy.save(os.path.join(os.getcwd(), self.model_name, "c"), self.c_list)
        numpy.save(os.path.join(os.getcwd(), self.model_name, "e"), self.e_list)
        plt.rcParams['font.size']=14
        x = [i for i in range(1,len(self.fitness_list))]
        y = [self.fitness_list[i]/min(self.fitness_list[i-1],self.popsize) for 
             i in range(1,len(self.fitness_list))]
        fig = plt.figure(figsize=(8,8))
        plt.scatter(x = x, y=y, c= [self.c_list[i] for i in range(1,
                                    len(self.fitness_list))], cmap="viridis")

        plt.ylabel("Growth rate", fontsize= 18)
        plt.xlabel("Time", fontsize= 18)
        
        fig.savefig(os.path.join(os.getcwd(), self.model_name, 
                 "fitness.png"))
        plt.close()
        if self.saving:
            numpy.save(os.path.join(os.getcwd(), self.model_name, self.model_name),
                       self.details_list)
            for i in range(5):
                fig = self.plot_over_time(i)
                fig[0].figure.savefig(os.path.join(os.getcwd(), self.model_name, 
                   str(i) + ".png"))
                plt.close()


'''functions'''              
def plot_rns(results_list, mean, sigma, gr = numpy.array([[3,1.25],[0,1.25]])):
    '''plots the evolved reaction norm shape
    
    results_list: list of suriviving genotypes; mean, sigma: mean and variance 
    of the environment (used for comparison with numerical calculations)'''
    
    mean_list = [numpy.mean(results_list[:,i]) for i in range(10)]
    sd_list = [numpy.std(results_list[:,i]) for i in range(10)]
    plt.errorbar(range(10), mean_list, sd_list, 
                 linestyle='None', marker='^')
    #for comparison: what is expected from calculation
    c = numpy.linspace(0,9,1000)
    y = stats.norm.cdf(c, mean, sigma)
    rn =[get_opt(gr, f, accuracy =100) for f in y]
    plt.plot(c,rn)

    plt.ylabel("Proportion P2", fontsize= 14)
    plt.xlabel("c", fontsize= 14)
    return(Genotype(mean_list).pars())

def get_opt(gr, f, accuracy = 1000):  #function comes from frontiers script
    '''finds the offspring proportion that maximises the geometric mean'''
    y = [geom(gr,f,p/accuracy) for p in range(accuracy +1)]
    n = numpy.array(y)
    opt = numpy.where(n==max(y))[0][0] / (len(y)-1)
    return(opt)
    
def geom (gr, f, p): #function comes from frontiers script
    '''calculates geometric mean fitness'''
    #gr = numpy.array([E1P1,E1P2],[E2P1,E2P2]])
    #f = frequency of environment E2
    #p = probability of P2
    
    GE1 = (1-p) * gr[0,0]  + p * gr[0,1]
    GE2 = (1-p) * gr[1,0]  + p * gr[1,1]
    g = GE1**(1-f)  * (GE2)**f
    return (g)
    
startpop = [Genotype([numpy.random.uniform(0,1) for i in range(10)]) for i in range(1000)]
test = Run_Program(model_name = "sd3",startpop = startpop, max_year = 20, env = [5,3], saving = True)
#test.run()

'''Phenotype-environment mismatch'''
#bet-hedging should evolve under unpredictable conditions, and plasticity under predictable conditions.
#these models shuold be very similar to the numerical approximation in Joschinski & Bonte, 
#Frontiers in Ecology and Evolution

sigmas = [0.01,2,100,2]
model_list =[]
gr_list = [numpy.array([[3,1.25],[0,1.25]]),numpy.array([[3,1.25],[0,1.25]]),
           numpy.array([[3,1.25],[0,1.25]]),numpy.array([[3,1.25],[0.5,1.25]])]

for i in range(4):
    model_list.append(Run_Program(model_name = "model"+str(i)+"-sd_"+str(sigmas[i]), startpop = startpop,
              env = [4.5, sigmas[i]],max_year = 4000, gr = gr_list[i]))
    model_list[i].run()
    model_list[i].save_data()

#data is saved here so that it does not have to be run again in case something in 
#output changes
    

#for each model:
dirs = ["E:/model_bh_plast/" + "model"+str(i)+"-sd_"+str(sigmas[int(i)]) + 
        "/results.npy" for i in range (4)]
for i in range (4):
    res = numpy.load(dirs[i]) 
    fig = plt.figure()
    plot_rns(res, 4.5, sigmas[i], gr = gr_list[i])
    fig.savefig("Fig1_"+str(i))


'''phenotypic variance costs'''
#we now assume that either canalization or phenotypic variance incurs a cost. If 
#phenotypic variance is ocstly, it does not matter whether the variance is due to
#plasticity or due to diversified bet-hedging
costtype = ["m","p1", "p2","e1","e2"]
costs =[-0.15, 0.15, 0.5]
model_list = []
for i in range (5):
    for j in range(3):
        x = Run_Program(model_name = "pvar"+ "-" + str(costtype[i])+
            "-mag_"+ str(costs[j]),
            startpop = startpop, env = [4.5, 2],max_year = 4000,
            gr = numpy.array([[3,1.25],[0,1.25]]),
            costs = "s", costtype = costtype[i], costmag = costs[j])
        model_list.append(x)
        x.run()
        x.save_data()

for i in range (1):
    for j in range(3):
        direc = "E:/model_bh_plast/" + "pvar-" + str(costtype[i]) +"-mag_" + str(
                costs[j]) + "/results.npy"
    
        res = numpy.load(direc) 
        fig = plt.figure()
        out = plot_rns(res, 4.5, 2, gr = numpy.array([[3,1.25],[0,1.25]]))
        fig.savefig("Fig2_"+str(i) + "-" + str(j))
        print ("parameters: ")
        print(out)
    
bh_init = Run_Program(model_name = "varcostsbh",
        startpop = startpop, env = [4.5, 100],max_year = 100,
        gr = numpy.array([[3,1.25],[0,1.25]]), 
        costs = "s", costmag =-0.075,costtype = "m")
bh_init.run()
bhmod = Run_Program(model_name = "varcostsbh",
        startpop = bh_init.eggs, env = [4.5, 100],max_year = 4000,
        gr = numpy.array([[3,1.25],[0,1.25]]), 
        costs = "s", costmag = -0.15,costtype = "m")

bhmod.run()
bhmod.save_data()
for i in range (2):
    for j in range(3):
        direc = "E:/model_bh_plast/" + "pvar-" + str(costtype[i]) +"-mag_" + str(
                costs[j]) + "/results.npy"
    
        res = numpy.load(direc) 
        arr = numpy.array([Genotype(res[i].tolist()).pars() for i in range(len(res))])
        print (numpy.mean(arr[1]))
        print (numpy.mean(arr[2]))
        print ("next\n")
        
 mean_list = [numpy.mean(results_list[:,i]) for i in range(10)]

'''plasticity costs'''
#needs rewrite
costs =[-0.2, 0.2, 0.5]
model_list = []
for i in range(3):
    model_list.append(Run_Program(model_name = "pcosts"+ str(i)+ "mag_"+ str(costs[i]),
        startpop = startpop, env = [4.5, 2],max_year = 4000,
        gr = numpy.array([[3,1.25],[0,1.25]]), costs = "r", costmag = costs[i]))
    model_list[i].run()
    model_list[i].save_data()

dirs = ["E:/model_bh_plast/" + "pcosts"+ str(i)+ "mag_"+ str(costs[i]) + 
        "/results.npy" for i in range (3)]    
for i in range (3):
    res = numpy.load(dirs[i]) 
    fig = plt.figure()
    plot_rns(res, 4.5, 2, gr = numpy.array([[3,1.25],[0,1.25]]))
    fig.savefig("Fig3_"+str(i))  
    