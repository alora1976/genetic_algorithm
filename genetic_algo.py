# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:07:18 2019

@author: Lori
"""

def fitness(Y,y):
    fitness=0
    for i in range(len(Y)):
        fitness+= (Y[i]- y[i])**2
    return fitness

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_points = 100
noise = np.random.normal(0,1,data_points)
x=np.random.uniform(0,20,data_points)
y=0.5*x+2+noise

plt.scatter(x,y)

df=pd.DataFrame({'x':x,'y':y})
df.to_csv('LineOfFit.csv')

def make_mutants (parent, num_mutants, step_size):
    mutants=[]
    for i in range (num_mutants):
        mutants.append([parent[0]+ np.random.normal (0, step_size),parent[1]+np.random.normal(0,step_size)])
        mutants.append(parent)
        return mutants
    
data_in=pd.read_csv('LineOfFit.csv', index_col=0)
x=data_in['x']
y=data_in['y']

max_generations=25
step_size=10
num_mutants=100

best_creature = [0,0]
Y=[ 0 for x_val in x]
best_score=fitness (Y,y)
last_score=best_score

mutant_list= make_mutants (best_creature, num_mutants,step_size)

creature_history=[]
score_history=[]
step_history=[]


for generation in range (0,max_generations):
    mutant_scores=[]
    for creature in mutant_list:
        Y=[creature[0]*x_val+creature[1] for x_val in x]
        mutant_scores.append(fitness(Y,y))
    best_score=min(mutant_scores)
    best_creature=mutant_list[mutant_scores.index(best_score)]
    
    creature_history.append(best_creature)
    score_history.append(step_size)
    
    if last_score==best_score:
        step_size=step_size/2
    
    last_score=best_score
    
    mutant_list=make_mutants(best_creature, num_mutants,step_size)
    
    print(best_creature)
