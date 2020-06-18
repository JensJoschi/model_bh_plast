# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

'''
In this model we consider a haploid genotype that faces the decision 
to produce one of two phenotypes (P1 and P2) under uncertain environmental 
conditions. A reaction norm determines the proportion of P2 in response to a cue c. 
The cue c is also related to an environment E, which can take two discrete states 
(E1 and E2). We model the probability of E2 as a logistic function to c,
p(c) = 1/(1 + exp(-k(c-c0)). The midpoint c0 thus determines the threshold of c 
that induces a switch towards E2 (i.e. the frequency of E2 across environmental cues),
while the slope k determines how well c predicts E2. 
The fitness of P1 is 4 in E1, but zero in E2, while the fitness of P2 is 1 
regardless of environmental conditions. 

A starting population has *popsize* individual genotypes that vary in reaction 
norm shapes to c. In each time step 5 environmental cues and according environments are 
created. Ten individuals per genotype and environment are allowed to chose phenotypes
according to their reaction norm shape and c, and their arithmetic mean fitness is
calculated. Then the geometric mean fitness of the genotype is calculated, based on 
average fitness in each of the 5 environments. The genotype reproduces with 
an offspring number equal to this geometric mean fitness. 

The offspring inherit the same genotype as their mother, except 
for the possibility of mutations with a rate of *mut_rate*. In the next time step 
the offspring replace the current population. If there are more than *popsize* offspring, 
*popsize* genotypes are randomly drawn from the seed bank. 

 The model runs for 2000 time steps, after which population size and reaction norm 
 parameters (Var_among, var_within, mean) are recorded.

explanation of var among + within needed. ref to preprint as soon as available

todo:
    midpoint makes little sense if rn is not logistic
    
it should be theoretically possible to only have 1 environment per time step, but 
unless the genotype is completely wrong, one of the 10 individuals often ends up
in a favorable environemnt and persists. So while there is a reasonable chance
for extinction, some genotypes get lucky and spread quickly, leading to high genotype
turnover but no evolution.

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

    def phenotype (self, c):
        return(bool(numpy.random.binomial(1,self.p_list[c],1)[0]))
        
    def pars(self):
        ''' calculate variance among and within environments

        var_within = sum p*(1-p) / n
        var_among = sd^2'''
        p = self.p_list
        p2 =[i * (1-i) for i in p]
 
        i = 0
        mp = "False" #gets midpoint of the reaction norm
        while mp == "False": # better would be "while not mp" but this causes a bug if mp becomes 0
            if p[i] >= max(p)/2:
                mp = i
            i = i + 1
        among = numpy.std(p)**2
        within = numpy.mean(p2) 
        return([mp, among, within]) 
 
        
        
class Run_Program(object):
    '''run the model'''
    def __init__ (self, model_name = "generic", max_year = 10000, saving =True, 
                  env = [5,3], startpop = [],
                  popsize = 500, mut_rate = 1/1000,
                  individuals = 10, environments = 5,
                  gr = numpy.array([[4,1],[0,1]])):

        '''saves parameters and creates starting population'''
        
        self.model_name = model_name
        self.max_year = max_year #number of time steps
        self.saving = saving #boolean: should details of reaction norms be saved at 
        #each time step?
        self.env = env #probability of E2 occuring: 
        #p(c) = 1/(1 + exp(-env[1] * (c-env[0])))
        #environments change on average at cue c = env[0]
        #env[1] determines whether the change in environments occurs always at env[0]
        #(env[1] infinitely high), or is spread across all c (env[1] small)
        self.startpop = startpop
        self.popsize = popsize #population size at start of the year
        self.mut_rate = mut_rate
        self.inds = individuals #replicates of the genotype
        self.environments = environments #number of environments to which the genotype 
        #is subjected in each time step
        self.gr = gr #growth rates of the 2 phenotypes in the 2 environemnts
        #sorting: [[E1 P1, E1 P2], [E2 P1, E2 P2]]
        #e.g. insect diapause: [[summer, nondiapausing = 3; summer diapasuing = 1],
        #[winter nondiapausing = 0, winter diapausing  = 1]]

        self.fitness_list = [] #stores no. survivors per time step
        self.details_list = [] #stores reaction norm shape at each time step
        
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
        '''wrapping method that runs the model, handles extinction, and saves output'''
        print("\nmodel: ", self.model_name, "\nRunning. 1 dot = 100 years")
       
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
        '''subject 10 instances of each genotype to 5 environments and calculate fitness'''
        
        def expon(x): #function to create Environment based on c
            return(1/(1+math.exp(-self.env[1] * (x-self.env[0]))))
            
        offspring_list = []
        for genotype in curr_list:
            geom_list =[] #stores arithmetic mean fitness in each of the 5 environments
            for i in range(self.environments):
                c = random.choice(range(10))
                E = numpy.random.binomial(1,expon(c),1)[0] #0 in summer, 1 in winter
                arm_list = [self.gr[E,int(genotype.phenotype(c))] for 
                                     k in range(self.inds)] #chooses phenotype 
                #for 10 individuals and calculates their arithmetic mean fitness
                geom_list.append(numpy.mean(arm_list))
            offspring_list.extend(genotype.reproduce(
                    numpy.exp(numpy.mean(numpy.log(geom_list))) #calculates geom
                    ))
        return(offspring_list)

    
    def save_rn (self, eggs):
        '''saves reaction norm parameters (var_within, var_among, ratio, sum, midpoint)'''
        x = [individual.pars() for individual in eggs]
        mp =[]
        among=[]
        within=[]
        ratio = []
        varsum =[]
        for i in x:
            mp.append(i[0])
            am = round(i[1],2)
            wi = round(i[2],2)
            among.append(am)
            within.append(wi)
            ratio.append(am/(am+wi) )
            varsum.append(am+wi)
        return([mp, among, within, ratio, varsum])
   
    def plot_over_time(self, variable=0):
        ylist = ["Midpoint", "Var_among", "Var_within", "Ratio", "Sum"]
        yl =[[0,10], [0,0.25], [0,0.25], [0,1], [0, 0.25]]
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
       ax.set_xlim3d(0,0.25)
       ax.set_ylim3d(0, 1)
       ax.set_zlim3d(0, 10)
       
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
    


startpop = [Genotype([numpy.random.uniform(0,1) for i in range(10)]) for i in range(500)]



all_results = []
#c0_list = [2.5, 3, 4.5]
c0_list = [4.5]
k_list = [0, 0.8 , 20]
P1_list = [0, 0.5, 1.5]
for i in c0_list:
    for j in k_list:
        for k in P1_list:
            row = []
            for l in range(1):
                 x = Run_Program(max_year = 2000, env = [i,j] , 
                            startpop = startpop, saving = True,
                            gr = numpy.array([[4,1],[k,1]]),                 
                            model_name = str(i) + "-" + str(j) + "-" + str(k)+"-" + str(l))
                 x.run()
                 x.save_data()
                 row.append(x)
            all_results.append(row)



models = [[0,3,6], [1,4,7], [2,5,8]]
xnames = ('unpredictable', 'intermediate', 'predictable')
groupnames =  ('high amplitude', 'intermediate', 'low amplitude')

def plotbarplot(models, xnames, groupnames, variable = 0):
    ylabs = ['Midpoint', 'variance among', 'variance within', "ratio", "sum"]
    N = len (models)
    nyear = len(all_results[0][0].details_list)
    ind = numpy.arange(N)
    width = 1/(1+N)
    data= []
    bars = []
    

    for i in range(N):
        temp = [numpy.mean(all_results[models[i][j]][0].details_list[nyear-1][variable]) for j \
                in range(len(models[i]))]
        temp2 = [numpy.std(all_results[models[i][j]][0].details_list[nyear-1][variable]) for j \
                in range(len(models[i]))]
        data.append(temp)
        bars.append(temp2)
    
    fig, ax = plt.subplots()
    
    w = 0
    group =[]
    for i in range(N):
        g = ax.bar(ind + w, tuple(data[i]), width, yerr = bars[i])
        group.append(g)
        w = w + width
    ax.set_ylabel(ylabs[variable])
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xnames)
    ax.legend(group, groupnames)
    
for i in range(20):
    plt.plot(all_results[0][0].eggs[random.choice(range(500))].p_list, 'b-')
    plt.plot(all_results[1][0].eggs[random.choice(range(500))].p_list, 'orange')
    plt.plot(all_results[2][0].eggs[random.choice(range(500))].p_list, 'green')
plt.title("Reaction norms in unpredictable environments")
plt.xlabel("Environmental cue c")
plt.ylabel ("Probability of P2")


for i in range(20):
    plt.plot(all_results[6][0].eggs[random.choice(range(500))].p_list, 'b-')
    plt.plot(all_results[7][0].eggs[random.choice(range(500))].p_list, 'orange')
    plt.plot(all_results[8][0].eggs[random.choice(range(500))].p_list, 'green')
plt.title("Reaction norms in predictable environments")
plt.xlabel("Environmental cue c")
plt.ylabel ("Probability of P2")



all_symmetric = []
#c0_list = [2.5, 3, 4.5]
c0_list = [4.5]
k_list = [0, 0.8 , 20]
P1_list = [0, 0.5, 1.5]
for i in c0_list:
    for j in k_list:
        for k in P1_list:
            row = []
            for l in range(1):
                 x = Run_Program(max_year = 2000, env = [i,j] , 
                            startpop = startpop, saving = True,
                            gr = numpy.array([[4,k],[k,4]]),                 
                            model_name = "symmetric_" +str(i) + "-" + str(j) + 
                            "-" + str(k)+"-" + str(l))
                 x.run()
                 x.save_data()
                 row.append(x)
            all_symmetric.append(row)
