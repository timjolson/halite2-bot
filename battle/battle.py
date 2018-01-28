import random
import subprocess
import json
import numpy as np
from tqdm import tqdm
import time
import os


# Print win records between matches
PRINT_PROGRESS = False


# read in ai bot file names
ais = os.listdir('./models')
ais = [ai for ai in ais if ai.endswith('.h5')]
# how many ai models
num_versions = len(ais)

# track wins per ai model
wins = [0]*num_versions
# track matches per ai model
num = [0]*num_versions

# run number of matches or until stop.txt exists (whichever happens first)
for m in tqdm(range(200)):
    
    if np.mod(m, 3)==0:
        files = os.listdir('.')
        for file in files:
            if file.endswith('.vec') or file.endswith('.log'): # or file.startswith('replay'):
                os.remove(file)
    
    # list of enemy bot files to choose from
    bot_list = [
        "../bots/tuned/mining.py",
        "../bots/tuned/mine_attack.py",
        "../bots/tuned/mine_attack.py",
        "../bots/tuned/mine_attack.py",
        "../bots/tuned/tuned_aggressive.py",
        "../bots/tuned/tuned_aggressive.py",
        "../bots/tuned/tuned_aggressive.py",
        "../bots/tuned/tuned_aggressive.py",
        "../bots/tuned/tuned_aggressive.py",
        "../bots/tuned/tuned_mine_attack.py",
        "../bots/tuned/tuned_mine_attack.py",
        "../bots/tuned/tuned_mine_attack.py",
        "../bots/tuned/tuned_mine_attack.py",
        "../bots/tuned/tuned_mine_attack.py"
    ]
    
    # build player list for the match (try until an ai player is chosen)
    while True:
        g2g = False  # whether there is an ai player in the match
        cmd = 'halite.exe'
        
        # store enemy type ('na' or ai version)
        enemy_version = []
        # list of ai versions to choose from
        ai_versions = list(range(num_versions))
        
        # add 2 or 4 players
        for _ in range(random.choice([2, 4])):
            # randomly choose from from ai or botlist
            enemy_version.append(random.choice(ai_versions+['na']*len(bot_list)))
            
            # build player into cmd string
            if enemy_version[-1] == 'na':
                # bot_list player
                cmd += ' "python {}"'.format(random.choice(bot_list))
            else:
                # ai player
                g2g = True
                ai_versions.remove(enemy_version[-1])
                cmd += ' "python ../bots/ai/MyBot{}.py"'.format(enemy_version[-1])
                num[enemy_version[-1]] += 1
        
        cmd += ' -q'
        
        if PRINT_PROGRESS:
            print('\n'+cmd)
        
        if g2g:
            # there's an ai player, end this while loop
            break
    
    # run match
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
    # translate results from json
    result = json.loads(result)
    
    # ignore matches where there's an error
    if not len(result['error_logs']):
        # sort players by rank
        ranked = sorted(result['stats'].items(), key=lambda t:t[1]['rank'])
        
        # get best ranked player
        for player in ranked:
            winner = int(player[0])
            winner = enemy_version[winner]
            break
        
        # if an ai won, append it's data to training files
        if winner != 'na':
            # get version into decimal format for file names
            version = round(winner/10,1)
            
            # get inputs
            with open("c{}_input.vec".format(version), "r") as f:
                input_lines = f.readlines()
            
            # get outputs
            with open("c{}_out.vec".format(version), "r") as f:
                output_lines = f.readlines()
                
            # check lengths, append to training data
            if len(input_lines)==len(output_lines):
                with open("train.in", "a") as f:
                    for l in input_lines:
                        f.write(l)
                with open("train.out", "a") as f:
                    for l in output_lines:
                        f.write(l)
            else:
                print(results)
                print(enemy_version)
                raise('Bot data lengths do not match')
                
            # track wins for winner
            wins[winner] += 1
        
        # print as we go
        if PRINT_PROGRESS:
            print('{} id {} beat out {}'.format(
                    winner, 
                    [p[0] for p in ranked if p[1]['rank']==1][0],
                    [e for e in enemy_version if e!=winner]
                    )
                )
            print(result['replay'])
            out = "Wins"
            for i in range(num_versions):
                if np.mod(i,8)==0:
                    out += '\n'
                out += '{}:{}/{}\t'.format(i,wins[i],num[i])
            print(out)
        
        if os.path.isfile('stop.txt'):
            break
    
    else:
        out = "***********\n***********\nWins"
        for i in range(num_versions):
            if np.mod(i,5)==0:
                out += '\n'
            out += '{}:{}/{}\t'.format(i,wins[i],num[i])
        print(out)
        
        print(result)
        print(enemy_version)
        print(result['replay'])
        break
    
    time.sleep(1)