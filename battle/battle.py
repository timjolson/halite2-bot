import random
import subprocess
import json
import numpy as np
from tqdm import tqdm
import time
import os

num_versions = len(os.listdir('./models'))
wins = [0]*num_versions

start = time.time()
num = [0]*num_versions

for _ in tqdm(range(200)):
    if os.path.isfile('stop.txt'):
        break
#    try: subprocess.Popen("clear.bat")
#    except FileNotFoundError: pass
    
    bot_list = [
        #"bot_mine/MyBot.py",
        "bot_mine_fill_attack/MyBot.py",
        "bot_mine_fill_attack/MyBot.py",
        "tuned_aggressive/MyBot.py",
        "tuned_aggressive/MyBot.py",
        "tuned_aggressive/MyBot.py",
        "tuned_mid_attack/MyBot.py",
        "tuned_mid_attack/MyBot.py",
        "tuned_mid_attack/MyBot.py",
        "tuned_mid_attack/MyBot.py",
        "aggressive/MyBot.py",
        "aggressive/MyBot.py",
        "aggressive/MyBot.py"
    ]
    while True:
        g2g = False
        num_players = 1
        cmd = 'halite.exe'
        enemy_version = []
        ai_versions = list(range(num_versions))
        for _ in range(random.choice([2, 4])):
            enemy_version.append(random.choice(ai_versions+['na']*12))
            num_players += 1
            if enemy_version[-1] == 'na':
                cmd += ' "python {}"'.format(random.choice(bot_list))
            else:
                g2g = True
                ai_versions.remove(enemy_version[-1])
                cmd += ' "python MyBot{}.py"'.format(enemy_version[-1])
                num[enemy_version[-1]] += 1
        cmd += ' -q -t'
        print('\n'+cmd)
        if g2g:
            break

    result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read()
    result = json.loads(result)

    if not len(result['error_logs']):
        ranked = sorted(result['stats'].items(), key=lambda t:t[1]['rank'])

        for player in ranked:
            winner = int(player[0])
            winner = enemy_version[winner]
            break

        if winner != 'na':
            version = round(winner/10,1)
            
            with open("c{}_input.vec".format(version), "r") as f:
                input_lines = f.readlines()
            with open("train.in", "a") as f:
                for l in input_lines:
                    f.write(l)

            with open("c{}_out.vec".format(version), "r") as f:
                output_lines = f.readlines()
            with open("train.out", "a") as f:
                for l in output_lines:
                    f.write(l)

            wins[winner] += 1
        
        print('{} id {} beat out {}'.format(
                winner, 
                [p[0] for p in ranked if p[1]['rank']==1][0],
                [e for e in enemy_version if e!=winner]
                )
            )
        print(result['replay'])
        out = "Wins \n"
        for i in range(num_versions):
            if np.mod(i,5)==0:
                out += '\n'
            out += '{}:{}/{}\t'.format(i,wins[i],num[i])
        print(out)

    time.sleep(1)