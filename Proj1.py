# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:32:18 2018

@author: Nicole Rita
"""

import json

path = 'C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT\\Project_One\\Beauty_5.json'
fopen = open(path, 'r')


'''
#Way to open .json given by the teacher
while 1:
    line = fopen.readline()
    sample = json.loads(line)
    print(sample) # Look at the different fields
    reviewText = sample['reviewText']
    fopen.close()
'''

#Opens json in another way
json_data = []
for line in open(path):
    json_data.append(json.loads(line))
    
    '''
cemMil = list()    
for i in range(100000):
    cemMil.append(json_data[i]['reviewText'])
    '''

#select random 100.000 rows
import random

num_to_select = 100000                    # Set the number to select here.
list_of_random_items = random.sample(json_data, num_to_select)

finalOneHundred = list()
for dictionary in list_of_random_items:
        finalOneHundred.append(dictionary.get('reviewText'))
        
 '''
for i in json_data:
    rand_reviews = [json_data[random.randrange(len(json_data))]
              for i in range(100000)]
    '''
       
        
        
        
        
        
        
        