# This file chooses 200 random files from the 1500 rear trucks zip file
# to create a training set

import random
import shutil
import os

source = '/Users/DanCassin/Desktop/Biggie/rear_truck_photos'
destination = '/Users/DanCassin/Desktop/Biggie/rear_truck_photos/training_photos' # folder must exist

total_trucks = os.listdir(source)

random_index = random.sample(range(0,1000), 200)

random_train_200 = [total_trucks[i] for i in random_index]

for file in random_train_200:
     new_path = shutil.move(f'{source}/{file}', destination)
     print(new_path)