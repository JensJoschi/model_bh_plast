# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

'''
In this model we consider a haploid genotype that faces the decision 
to produce one of two phenotypes (P1 and P2) under uncertain environmental 
conditions. A reaction norm determines the proportion of p2 in response to a cue c. 
An environment E can take two discrete states (E1 and E2); the probability 
of E2 is a logistic function of c, p(c) = 1/(1 + exp(-k(c-c0)). In this function the
midpoint c0 determines the frequency of E2, and the slope k determines how well the
cue c predicts the change in environments.
The fitness of P1 is 2.2 in E1, but zero in E2, while the fitness of P2 is 1 
regardless of environmental conditions. 

A starting population has *popsize* individual genotypes that vary in reaction 
norm shapes. At each time 5 environmental cues and according environments are 
created. Ten individuals per genotype and environment are allowed to chose phenotypes
according to their reaction norm shape and the cues, and the geometric mean
fitness across the 5 environments is calculated. The genotype then reproduces with 
an offspring number equal to the average geometric mean fitness of the 10 individuals. 

The offspring inherit the same genotype as their mother, except 
for the possibility of mutations with a rate of *mut_rate*. In the next time step 
the offspring replace the current population. If there are more than *popsize* offspring, 
*popsize* genotypes are randomly drawn from the seed bank. 

 The model runs for x time steps, after which population size and reaction norm 
 parameters (Var_among, var_within, mean) are recorded.

explanation of var among + within needed. ref to preprint as soon as available

todo:
    midpoint makes little sense if rn is not logistic
    
it should be theoretically possible to only have 1 environment per time step, but 
unless the genotype is completely wrong, one of the 10 individuals often ends up
in a favorable environemnt and persists. So while there is a reasonable chance
for extinction, some genotypes get lucky and spread quickly, leading to high genotype
turnover but no evolution.


something is wrong with gr
'''

import random
import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import copy
import os

'''classes'''
class Genotype(object):
    ''' Creates an individuum with a reaction norm shape determined by 10 parameters'''
    
    def __init__(self,p_list = []):
        self.p_list = p_list if p_list else [
                numpy.random.uniform(0,1) for i in range(10)]
        
    def __str__(self):
        return(str(self.p_list)) 
    
    def reproduce(self, r): 
        '''reproduces the genotype r times
        
        r = 4.2 will return a list with 4 individuals in 80% of all cases, and with 5
        individuals in the remaining 20% of all cases. expected input: r = float > 0'''
        n = math.floor(r) + numpy.random.binomial(1, r- math.floor(r),1)[0]
        return( [copy.deepcopy(self) for i in range(0,n)] )  
             
    def mutate(self, mut_frac):
       '''induces mutations in the genotype's reaction norm'''
         
       self.p_list = [i if (random.uniform(0,1) > mut_frac) else 
                      random.uniform(0,1) for i in self.p_list]
       #alternative: random.choice([0,random.uniform(0,1),1]) for i in self.p_list]
       #mutations make each loci 0, 1, or a random draw between 0 and 1
    def phenotype (self, c):
        return(bool(numpy.random.binomial(1,self.p_list[c],1)[0]))
        
    def pars(self):
        ''' calculate variance among and within environments

        var_within = sum p*(1-p) / n
        var_ax.mong = sd^2 / n'''
        p = self.p_list
        p2 =[i * (1-i) for i in p]
 
        i = 0
        mp = "False" #gets midpoint of the reaction norm
        while mp == "False": # better would be "while not mp" but this causes a bug if mp becomes 0
            if p[i] >= max(p)/2:
                mp = i
            i = i + 1
        among = numpy.std(p, ddof=1)
        within = numpy.mean(p2) 
        return([mp, among, within]) 
 
        
        
class Run_Program(object):
    '''run the model'''
    def __init__ (self, model_name = "generic", max_year = 10000, saving =True, 
                  env = [5,3], startpop = [],
                  popsize = 1000, mut_rate = 1/1000,
                  individuals = 10, environments = 5,
                  gr = numpy.array([[3,1],[0,1]])):
        #correpsonds to [[environment], [phenotype]]
        #for insect diapause: summer, nondiapausing = 3; summer diapasuing = 1
        #winter nondiapausing = 0, winter diapausing  = 1
                 
        self.testlist=[]
        '''saves parameters and creates starting population'''
        
        self.model_name = model_name
        self.max_year = max_year
        self.saving = saving
        self.env = env #probability of E2 occuring: 
        #p(c) = 1/(1 + exp(-env[1] * (c-env[0])))
        #environments change on average at cue c = env[0]
        #env[1] determines whether the change in environments occurs always at env[0]
        #(env[1] infinitely high), or is spread across all c (env[1] small)
        self.startpop = startpop
        self.popsize = popsize #population size at start of the year
        self.mut_rate = mut_rate
        self.inds = individuals #replicates of the genotype
        self.environments = environments
        self.gr = gr


        self.fitness_list = []
        self.details_list = []
        
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
        fourth_line = str('individuals: {:2}, environments: {:2}'.format(
                self.inds, self.environments))
        return(first_line+ second_line+ third_line+ fourth_line)
        
        
    def run(self):
        print("\nmodel: ", self.model_name, "\nRunning. 1 dot = 100 years")
       
        yc = 1
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
                    self.details_list.append(self.save_rn(
                            random.sample(self.eggs,min(len(self.eggs),100))
                            ))
                yc +=1
                if not yc % 100: 
                    print(".", end = '')
            elif extinct_at:
                pass
            else:
                extinct_at = yc
        if extinct_at:
            print ("Population extinct in year", extinct_at)
        self.results = numpy.array(self.save_rn(self.eggs))
        if self.saving:
            self.details_list = numpy.array(self.details_list)
        return(yc)
        
    def runyear(self, curr_list):
        '''docstring needed'''
        def expon(x):
            return(1/(1+math.exp(-self.env[1] * (x-self.env[0]))))
        offspring_list = []
        for genotype in curr_list:
            geom_list =[]
            for i in range(self.environments):
                c = random.choice(range(10))
                E = numpy.random.binomial(1,expon(c),1)[0] #0 in summer, 1 in winter
                arm_list = [self.gr[E,int(genotype.phenotype(c))] for 
                                     k in range(self.inds)] #chooses phenotype 
                #for 10 individuals and calculates their arithmetic mean 
                #fitness
               # print (arm_list)
                geom_list.append(numpy.mean(arm_list)) # puts 
                # arithmetic mean of these 10 individuals into a list to calc their 
                #geom across the environments
            #print(geom_list)
           # print("Fitness: ", numpy.exp(numpy.mean(numpy.log(geom_list))))
            offspring_list.extend(genotype.reproduce(
                    numpy.exp(numpy.mean(numpy.log(geom_list))) #calculates geom
                    ))
            self.testlist.append(numpy.exp(numpy.mean(numpy.log(geom_list))))
            #print ("total", len(offspring_list),"\n")
        return(offspring_list)

    
    def save_rn (self, eggs):
        x = [individual.pars() for individual in eggs]
        mp =[]
        among=[]
        within=[]
        for i in x:
            mp.append(i[0])
            among.append(round(i[1],2))
            within.append(round(i[2],2))
        return([mp, among, within])
   
    def save_data(self):
        os.mkdir(self.model_name)
        numpy.save(os.path.join(os.getcwd(), self.model_name, self.model_name),
                       self.fitness_list) #model name twice to make generic.npy in subfolder "generic"
        numpy.save(os.path.join(os.getcwd(), self.model_name, "startpop"), 
                       self.startpop)
        fig = self.plot_all(3)
        fig[0].figure.savefig(os.path.join(os.getcwd(), self.model_name, 
                 "fitness.png"))
        if self.saving:
            numpy.save(os.path.join(os.getcwd(), self.model_name, self.model_name),
                       self.details_list)
            for i in range(3):
                fig = self.plot_all(i)
                fig[0].figure.savefig(os.path.join(os.getcwd(), self.model_name, 
                   str(i) + ".png"))
                

'''functions'''

def plot_3d (rn):
       m = rn[0] #mean
       s = rn[1]+rn[2] # sum, phenotypic variance
       r = rn[1]/(rn[1]+rn[2])#ratio of var components
       
       xyz = numpy.vstack([r,s,m])
                  
       cm = plt.get_cmap("viridis")
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.set_xlabel('Variance composition(r)')
       ax.set_ylabel('Phenotypic variance(s)')
       ax.set_zlabel('Midpoint(m)')
       ax.set_xlim3d(0, 1)
       ax.set_ylim3d(0, 0.78)
       ax.set_zlim3d(0, 10)
       #ylim: among = 0.527, within = 0.25
       try:
           density = stats.gaussian_kde(xyz)(xyz)
           idx = density.argsort()
           x, y, z, density = r[idx], s[idx], m[idx], density[idx]
           ax.scatter(x, y, z, c=density, cmap = cm)
       except:
           ax.scatter(r,s,m)
       plt.show()
        
def plot_summary(model_array, variable = 0):
    for i in range(len(model_array)):
        res = model_array[i].results
        val_y = res[variable]
        val_n = res[variable+3]
        plt.scatter(numpy.full(len(val_y),i), val_y, 
                c = sum(val_n)/val_n, cmap = "Blues_r", s = 10, marker="s")
    

def plot_over_time(model, variable=0):
    mean = [numpy.mean(model.details_list[i][variable]) for i in range(len(
            model.details_list))]
    sd = [numpy.std(model.details_list[i][variable]) for i in range(len(
            model.details_list))]
    plt.plot(mean)



startpop = [Genotype([numpy.random.uniform(0,1) for i in range(10)]) for i in range(1000)]
# startpop adjusted due to high risk for extinction 

plasticity = Run_Program(max_year = 1000, env = [4.5,20], startpop = startpop)
#plasticity.run()
bethedging = Run_Program(max_year = 10000, env = [4.5,1], startpop = startpop)
#bethedging.run() 

c1  = Genotype([0 for i in range(10)]).reproduce(1000) #growth throughout season
c2  = Genotype([1 for i in range(10)]).reproduce(1000) #instant diapause
dbh = Genotype([0.5 for i in range(10)]).reproduce(1000) #50% of each under all conditions
pp  = Genotype([0,0,0,0,0,1,1,1,1,1]).reproduce(1000) # diapause timing

print ("strategies in summer")
c1_in_summer = Run_Program(max_year = 50, env = [12, 20], mut_rate = 0, 
                                startpop = c1, model_name ="grower")
c2_in_summer = Run_Program(max_year = 50, env = [12, 20], mut_rate = 0, 
                                startpop = c2, model_name ="diapauser")
plastic_in_summer = Run_Program(max_year = 50, env = [12, 20], mut_rate = 0, 
                                startpop = pp, model_name ="plastic, summer")
dbh_in_summer= Run_Program(max_year = 50, env = [12, 20], mut_rate = 0, 
                            startpop = dbh, model_name = "dbh, summer")


c1_in_summer.run() #constant at 3000
c2_in_summer.run() #constant at 1000
plastic_in_summer.run() # constant at ~ 1800
dbh_in_summer.run() #constant at ~ 2000


print ("strategies in winter")
c1_in_winter = Run_Program(max_year = 50, env = [0, 20], mut_rate = 0, 
                                startpop = c1, model_name ="grower")
c2_in_winter = Run_Program(max_year = 50, env = [0, 20], mut_rate = 0, 
                                startpop = c2, model_name ="diapauser")

c1_in_winter.run() #instant extinction
c2_in_winter.run() #constant at 1000

print ("strategies in predictable climates")
c1_in_predictable = Run_Program(max_year = 50, env = [4.5, 20], mut_rate = 0, 
                                startpop = c1, model_name ="grower")
c2_in_predictable = Run_Program(max_year = 50, env = [4.5, 20], mut_rate = 0, 
                                startpop = c2, model_name ="diapauser")
plastic_in_predictable = Run_Program(max_year = 50, env = [4.5, 20], mut_rate = 0, 
                                startpop = pp, model_name ="plastic, predictable")
dbh_in_predictable = Run_Program(max_year = 50, env = [4.5, 20], mut_rate = 0, 
                            startpop = dbh, model_name = "dbh, predictable")

c1_in_predictable.run() #nearly instant extinction
c2_in_predictable.run() #constant at 1000
plastic_in_predictable.run() #constant at  ~ 1800
dbh_in_predictable.run() #constant at just above 1000

print ("strategies in variable climates")
c1_in_variable = Run_Program(max_year = 100, env = [4.5, 0], mut_rate = 0, 
                                startpop = c1, model_name ="grower")
c2_in_variable = Run_Program(max_year = 100, env = [4.5, 0], mut_rate = 0, 
                                startpop = c2, model_name ="diapauser")
plastic_in_variable = Run_Program(max_year = 100, env = [4.5, 0], mut_rate = 0, 
                                startpop = pp, model_name ="plastic, variable")
dbh_in_variable = Run_Program(max_year = 100, env = [4.5, 0], mut_rate = 0, 
                            startpop = dbh, model_name = "dbh, variable")

c1_in_variable.run() #instant extinction
c2_in_variable.run() #constant at 1000
plastic_in_variable.run() #instant extinciton
dbh_in_variable.run() #constant at just above 1000; this means that dbh and 
#cbh provide nearly the same fitness

print ("mix in variable")
specmix = Genotype([0 for i in range(10)]).reproduce(500)
specmix.extend(Genotype([1 for i in range(10)]).reproduce(500))

specmix_in_variable = Run_Program(max_year = 100, env = [4.5, 0], mut_rate = 0, 
                                startpop = specmix, model_name ="grower")
specmix_in_variable.run() #rapid drop to 500 (cbh only)

dbhpp = Genotype([0.5 for i in range(10)]).reproduce(500)
dbhpp.extend(Genotype([0,0,0,0,0,1,1,1,1,1]).reproduce(500))

comp = Run_Program(max_year = 1000, env = [4.5, 0], mut_rate = 0, 
                                startpop = dbhpp, model_name ="grower")
comp.run() #rapid increase to just above 1000 (bethedgers only)

'''
n = 10000
outcomes = []
for i in range(5):
    outcomes.append(Run_Program(max_year = n, saving = True, model_name = "sigma"+
                                str(i/2), winter_list = make_climate(mu = 15,
                                        sigma =i/2, n = n), gr_awake=3))
    #outcomes[i].run()

outcomes[0] = Run_Program(max_year = n, saving = True, model_name = "sigma_inf",
                                winter_list = make_climate(mu = 15,
                                        sigma =5, n = n), gr_awake=3)
outcomes[0].run()
#outcomes[4].run()
#plot_3d(outcomes[0].details_list[0])
plot_over_time(outcomes[0])
plot_over_time(outcomes[1])
plot_over_time(outcomes[2])
plot_over_time(outcomes[3])
plot_over_time(outcomes[4])
plt.title("Evolution of midpoint")
plt.legend([0,0.5,1,1.5,2])


plot_over_time(outcomes[0],1)
plot_over_time(outcomes[1],1)
plot_over_time(outcomes[2],1)
plot_over_time(outcomes[3],1)
plot_over_time(outcomes[4],1)
plt.title("Evolution of var_among")
plt.legend([0,0.5,1,1.5,2])


'''


'''
l = []
for i in range(1000):
    rn = Genotype().pars()
    m = rn[0] #mean
    s = rn[1]+rn[2] # sum, phenotypic variance
    r = rn[1]/(rn[1]+rn[2])#ratio of var components
    l.append([m,s,r])
    


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Variance composition(r)')
ax.set_ylabel('Phenotypic variance(s)')
for i in range(len(l)):
    ax.scatter(l[i][1],l[i][2],l[i][0])
ax.set_zlabel('Midpoint(m)')

results = numpy.array(test.details_list)
r = numpy.mean(results, axis=2)
plt.plot(r[:,0])



test.details_list[0]
n=[]
for i in range(len(test.details_list)):
    n.append(numpy.argmax(test.details_list[i][3]))
    
m = [test.details_list[i][0][n[i]] for i in range(len(test.details_list))]

n=[]
for i in range(len(test.details_list)):
    n.append(numpy.argmax(test.details_list[i][4]))
a = [test.details_list[i][1][n[i]] for i in range(len(test.details_list))]


variable =0
for i in range(len(test.details_list)):
    results = test.details_list[i]
    val_y = results[variable]
    val_n = results[variable+3]
    plt.scatter(numpy.full(len(val_y),i), val_y, 
                c = sum(val_n)/val_n, cmap = "Blues_r", s = 10, marker="s")
'''
