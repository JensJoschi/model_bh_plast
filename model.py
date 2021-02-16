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
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import stats
import copy
import os

'''classes'''
class Genotype(object):
    ''' Creates an individuum with a reaction norm shape determined by 10 parameters'''
    
    def __init__(self,p_list = [], costs = [], costmag = []):
        self.p_list = p_list if p_list else [
                numpy.random.uniform(0,1) for i in range(10)]
    
        
    def __str__(self):
        return(str(self.p_list)) 
    
    def reproduce(self, r, costs = [], costmag = []): 
        '''reproduces the genotype r times
        
        r = 4.2 will return a list with 4 individuals in 80% of all cases, and with 5
        individuals in the remaining 20% of all cases. expected input: r = float > 0'''
        if costs: #for the moment these are only phenotypic variance costs
            t = self.pars()
            s = (t[1]+t[2])/0.25 #makes phenotypic variance go from 0-1            
            r = r + r * costmag * s #positive costmag = benefit, negative = cost
        n = math.floor(r) + numpy.random.binomial(1, r- math.floor(r),1)[0]
        return( [copy.deepcopy(self) for i in range(0,n)] )  
             
    def mutate(self, mut_frac):
       '''induces mutations in the genotype's reaction norm'''
         
       self.p_list = [i if (random.uniform(0,1) > mut_frac) else 
                      random.uniform(0,1) for i in self.p_list]

    def phenotype (self, c):
        return(bool(numpy.random.binomial(1,self.p_list[c],1)[0]))
        
    def pars(self):
        ''' calculate variance among and within environments

        var_within = sum p*(1-p) / n
        var_among = sd^2'''
        p = self.p_list
        p2 =[i * (1-i) for i in p]
        f = sum(p)/10
        among = numpy.std(p)**2
        within = numpy.mean(p2) 
        return([f, among, within]) 
    

        
        
class Run_Program(object):
    '''run the model'''
    def __init__ (self, model_name = "generic", max_year = 1000, saving =False, 
                  env = [5,3], startpop = [],
                  popsize = 3000, mut_rate = 1/1000,
                  replicates = 100,
                  gr = numpy.array([[4,1],[0.1,1]]),
                  costs = [], costmag = []):

        '''saves parameters and creates starting population'''
        
        self.model_name = model_name
        self.max_year = max_year #number of time steps
        self.saving = saving #boolean: should details of reaction norms be saved at 
        #each time step?
        self.env = env #mean and variance of the environment
        self.startpop = startpop
        self.popsize = popsize #population size at start of the year
        self.mut_rate = mut_rate
        self.replicates = replicates #replicates of the genotype
        self.gr = gr #growth rates of the 2 phenotypes in the 2 environemnts
        #sorting: [[E1 P1, E1 P2], [E2 P1, E2 P2]]
        #e.g. insect diapause: [[summer, nondiapausing = 3; summer diapasuing = 1],
        #[winter nondiapausing = 0, winter diapausing  = 1]]
        self.costs = costs #for the moment only phenotypic variance costs; later costs should be
        #[], "s","r" or similar. negative costs are costs of phenotypic variance, positive
        #values are benefits of variance/costs of canalization
        self.costmag = costmag #can later be used for the magnitude of any cost now
        #only phenotypic variance costs
        
        
        self.fitness_list = [] #stores no. survivors per time step
        self.details_list = [] #stores reaction norm shape at each time step
        self.sample_list = [] #stores a sample of 100 genotypes in each time step
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
                if self.saving:
                    self.sample_list.append(random.sample(self.eggs,min(len(self.eggs),100)))
                    self.details_list.append(self.save_rn(
                            random.sample(self.eggs,min(len(self.eggs),100))
                            ))
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
        self.results = numpy.array(self.save_rn(self.eggs))
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
        e = stats.norm.cdf(c, self.env[0], self.env[1])
        E = numpy.random.binomial(1,e,1)[0] #0 in summer, 1 in winter   
        #these paramters are the same for every replicate and for every genotype!
        offspring_list = []
        for genotype in curr_list:
            replicate_fitness = [self.gr[E,int(genotype.phenotype(c))] for r in range(self.replicates)]            
            offspring_list.extend(genotype.reproduce(numpy.mean(replicate_fitness), 
                                                     self.costs, self.costmag))
        return(offspring_list)

    
    def save_rn (self, eggs):
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
   
    def plot_over_time(self, variable=0):
        ylist = ["Frequency P2", "Var_among", "Var_within", "Ratio", "Sum"]
        yl =[[0,1], [0,0.25], [0,0.25], [0,1], [0, 0.25]]
        mean = [numpy.mean(self.details_list[i][variable]) for i in range(len(
            self.details_list))]
        sd = [numpy.std(self.details_list[i][variable]) for i in range(len(
            self.details_list))]
        upper = [mean[i]+sd[i] for i in range(len(mean))]
        lower = [mean[i]-sd[i] for i in range(len(mean))]
        plt.rcParams['font.size']=14
        Fig = plt.plot(mean, 'ks-', linewidth = 2.0)
        plt.ylabel(ylist[variable], fontsize= 14)
        plt.ylim(yl[variable])
        plt.xlabel("Time", fontsize= 14)
        plt.plot(upper, color = "grey", linestyle = "dashed")
        plt.plot(lower, color = "grey", linestyle = "dashed")
        return (Fig)

    def save_data(self):
        os.mkdir(self.model_name)
        numpy.save(os.path.join(os.getcwd(), self.model_name, self.model_name),
                       self.fitness_list) #model name twice to make generic.npy in subfolder "generic"
        numpy.save(os.path.join(os.getcwd(), self.model_name, "startpop"), 
                       self.startpop)
        numpy.save(os.path.join(os.getcwd(), self.model_name, "results"), 
                       self.results)
        plt.rcParams['font.size']=14
        fig = plt.plot(self.fitness_list, 'bs-.', linewidth = 2.0)
        plt.ylabel("Population size", fontsize= 18)
        plt.xlabel("Time", fontsize= 18)
        
        fig[0].figure.savefig(os.path.join(os.getcwd(), self.model_name, 
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
                

startpop = [Genotype([numpy.random.uniform(0,1) for i in range(10)]) for i in range(1000)]


'''Phenotype-environment mismatch'''
#bet-hedging should evolve under unpredictable conditions, and plasticity under predictable conditions.
#these models shuold be very similar to the numerical approximation in Joschinski & Bonte, 
#Frontiers in Ecology and Evolution

test = Run_Program(model_name = "sd3",startpop = startpop, max_year = 100, env = [5,3], saving = True)
test.run()
test2 = Run_Program(model_name = "sd3",startpop = test.eggs, max_year = 400, env = [5,3], saving = True)
#test2.run()
test3 = Run_Program(model_name = "sd3",startpop = test2.eggs, max_year = 500, env = [5,3], saving = True)
#test3.run()

sigmas = [0.01,2,4,6,8,10,100]
model_list =[]
for i in range(6):
    model_list.append(Run_Program(model_name = "sd_"+str(sigmas[i]), startpop = startpop,
              env = [5, sigmas[i]]))
    model_list[i].run()
    model_list[i].save_data()

#figure: change in mean frequencies with sd
for i in range(6):
    plt.plot(sigmas[i], numpy.mean(model_list[i].results[0]),  "bo")

for i in range(6):
    plt.plot( sigmas[i],numpy.mean(model_list[i].results[3]), "bo")
 
           
'''phenotype costs'''
#here we assume that one of the phenotypes is more costly to produce, i.e. the growth rate
#of either P1 or P2 is lowered. 

#scenario #1: phenotype P2 is more costly to produce
pc1_0 = Run_Program(model_name = "phenotype-costs_plastic",startpop = startpop, 
        max_year = 1000, env = [5,0.01], saving = True, gr = numpy.array([[4,0.7],[0.1,0.7]]))
pc1_5 = Run_Program(model_name = "phenotype-costs_dbh",startpop = startpop, 
        max_year = 1000, env = [5,10], saving = True,gr = numpy.array([[4,0.7],[0.1,0.7]]))

#scenario #2: P2 is more costly, but only in one environement

pc2_0 = Run_Program(model_name = "phenotype-costs2_plastic",startpop = startpop, 
        max_year = 1000, env = [5,0.01], saving = True, gr = numpy.array([[4,0.7],[0.1,1]]))
pc2_5 = Run_Program(model_name = "phenotype-costs2_dbh",startpop = startpop, 
        max_year = 1000, env = [5,10], saving = True,gr = numpy.array([[4,0.7],[0.1,1]]))

#scenario #3: costs in the other environment
pc3_0 = Run_Program(model_name = "phenotype-costs3_plastic",startpop = startpop, 
        max_year = 1000, env = [5,0.01], saving = True, gr = numpy.array([[4,1],[0.1,0.7]]))
pc3_5 = Run_Program(model_name = "phenotype-costs3_dbh",startpop = startpop, 
        max_year = 1000, env = [5,10], saving = True,gr = numpy.array([[4,1],[0.1,0.7]]))

#scenario 4: P1 is more costly (only one env here because it is already at zero in the other one)
p4_0 = Run_Program(model_name = "phenotype-costs4_plastic",startpop = startpop, 
        max_year = 1000, env = [5,0.01], saving = True, gr = numpy.array([[2.5,1],[0.1,1]]))
p4_5 = Run_Program(model_name = "phenotype-costs4_dbh",startpop = startpop, 
        max_year = 1000, env = [5,10], saving = True, gr = numpy.array([[2.5,1],[0.1,1]]))




'''phenotypic variance costs'''
#we now assume that either canalization or phenotypic variance incurs a cost. If 
#phenotypic variance is ocstly, it does not matter whether the variance is due to
#plasticity or due to diversified bet-hedging
test2 = Run_Program(model_name = "costs",startpop = startpop, max_year = 100,
                    env = [5,3], saving = True, costs = "lololo", costmag = -0.5)
test2.run() #should go extinct nearly instantly
test3 = Run_Program(model_name = "costs",startpop = startpop, max_year = 100, 
                    env = [5,3], saving = True, costs = "lololo", costmag = -0.2)
test3.run() #compare this to test further up. looks like canalization evolves

'''functions'''
'''these things are currently not in use'''
def plot_3d (rn):
       f = rn[0] #mean
       s = rn[1]+rn[2] # sum, phenotypic variance
       r = rn[1]/(rn[1]+rn[2])#ratio of var components
       
       xyz = numpy.vstack([r,s,f])
                  
       cm = plt.get_cmap("viridis")
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.set_xlabel('Variance composition(r)')
       ax.set_ylabel('Phenotypic variance(s)')
       ax.set_zlabel('Frequency P2 (f)')
       ax.set_xlim3d(0,0.25)
       ax.set_ylim3d(0, 1)
       ax.set_zlim3d(0, 10)
       
       try:
           density = stats.gaussian_kde(xyz)(xyz)
           idx = density.argsort()
           x, y, z, density = r[idx], s[idx], f[idx], density[idx]
           ax.scatter(x, y, z, c=density, cmap = cm)
       except:
           ax.scatter(r,s,f)
       plt.show()
        
def plot_summary(model_array, variable = 0):
    for i in range(len(model_array)):
        res = model_array[i].results
        val_y = res[variable]
        val_n = res[variable+3]
        plt.scatter(numpy.full(len(val_y),i), val_y, 
                c = sum(val_n)/val_n, cmap = "Blues_r", s = 10, marker="s")
    






all_results = []
c0_list = [4.5] #make 3, 4.5, 6
k_list = [0, 1, 2]
gr_list = []
for i in [2,3,4]:
        for j in [0, 0.5, 1.5]:
            gr_list.append(numpy.array([[i,1],[j,1]]))
N = 1000
parm_list =[]
for midpoints in c0_list:
    for predictabilities in k_list:
        for growth_rates in gr_list:
            row = []
            for replicates in range(1):
                 x = Run_Program(max_year = N, env = [midpoints,predictabilities] , 
                            startpop = startpop, saving = True,
                            gr = growth_rates,                 
                            model_name = str(midpoints) + "-" + str(predictabilities) + 
                            "-" + str(growth_rates[0,0])+"_" + str(growth_rates[1,0])+
                            "-"+ str(replicates))
                 x.run()
                 x.save_data()
                 row.append(x)
                 parm_list.append([midpoints,predictabilities, growth_rates])
            all_results.append(row)
