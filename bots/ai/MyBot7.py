# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
import logging, time, os, random, subprocess, sys
import numpy as np
from hlt.utils import Struct
from hlt.constants import *

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
#os.environ["CUDA_VISIBLE_DEVICES"] = ''
import keras
import tensorflow as tf
from keras.models import load_model

tf.logging.set_verbosity(tf.logging.ERROR)

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
sys.stderr = stderr

VERSION = 7
models = os.listdir('./models')
model = load_model('./models/'+models[VERSION])
VERSION = 0 + np.round(VERSION/10, 1)

#time.sleep(1)
try: subprocess.Popen("del -f c{}_input.vec".format(VERSION))
except FileNotFoundError: pass
try: subprocess.Popen("del -f c{}_out.vec".format(VERSION))
except FileNotFoundError: pass

COS_LOOKUP = {a:8*np.cos(np.deg2rad(a)) for a in np.arange(0,360,1)}
SIN_LOOKUP = {a:8*np.sin(np.deg2rad(a)) for a in np.arange(0,360,1)}

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("v{}".format(VERSION), logging.INFO)

try:
    me, turn, ran_once = None, 0, False

    while True:
        # Update the map for the new turn and get the latest version
        game_map = game.update_map()
        start = time.time()
        
        if not ran_once:
            me = game_map.get_me()
            num_players = len(game_map.all_players())
            area = game.map.width * game.map.height
            logging.info("Starting, my id: %d"%(me.id))
            ran_once = True

        #logging.info('Start turn {} @ {}'.format(turn, time.time()-start))
        S = Struct(all=Struct(),my=Struct(),their=Struct())
        S.my.all = game_map.get_me().all_ships()
        S.my.docked = [s for s in S.my.all if s.docking_status != s.DockingStatus.UNDOCKED]
        S.my.undocked = [s for s in S.my.all if s.docking_status == s.DockingStatus.UNDOCKED]
        S.their.all = game_map.their_ships()
        S.their.docked = [s for s in S.their.all if s.docking_status != s.DockingStatus.UNDOCKED]
        S.their.undocked = [s for s in S.their.all if s.docking_status == s.DockingStatus.UNDOCKED]
        S.all = S.my.all + S.their.all
        # logging.info('ALL {}'.format(S.my.all))
        # logging.info('DOCKED {}'.format(S.my.docked))
        # logging.info('UNDOCKED {}'.format(S.my.undocked))

        P = Struct(all=Struct(), my=Struct(), their=Struct())
        P.all = game_map.all_planets()
        P.my.all = [p for p in P.all if p.owner is not None and p.owner.id == me.id]
        P.my.dockable = [p for p in P.my.all if p.ratio_docked < 1.0]
        P.fresh = [p for p in P.all if p.owner is None]
        P.dockable = [p for p in P.fresh+P.my.dockable]
        P.their = [p for p in P.all if p not in P.my.all+P.fresh]

        C = Struct(
            ships=Struct(my=Struct(), their=Struct()),
            planets=Struct(my=Struct(), their=Struct())
        )
        C.ships.all = len(S.all)
        C.ships.my.all = len(S.my.all)/C.ships.all
        C.ships.my.docked = len(S.my.docked)/C.ships.all
        C.ships.my.undocked = len(S.my.undocked)/C.ships.all
        C.ships.their.all = len(S.their.all)/C.ships.all
        C.ships.their.docked = len(S.their.docked)/C.ships.all
        C.ships.their.undocked = len(S.their.undocked)/C.ships.all
        C.planets.all = len(P.all)
        C.planets.my.all = len(P.my.all)/C.planets.all
        C.planets.my.dockable = len(P.my.dockable)/C.planets.all
        C.planets.fresh = len(P.fresh)/C.planets.all
        C.planets.their = len(P.their)/C.planets.all
        C.planets.dockable = len(P.dockable)/C.planets.all

        D = Struct(
            planets = Struct(
                all = 300*np.sum([p.radius for p in P.all])/area,
                my = 300*np.sum([p.radius for p in P.my.all])/area,
                their = 300*np.sum([p.radius for p in P.their])/area,
                fresh = 300*np.sum([p.radius for p in P.fresh])/area,
                dockable = 300*np.sum([p.radius for p in P.dockable])/area
            ),
            ships = Struct(
                all = 20000*C.ships.all/area,
                my = 20000*C.ships.my.all/area,
                their = 20000*C.ships.their.all/area
            )
        )
        # logging.info('filter end={}'.format(time.time() - start))

        input_vector = [num_players, area/98304, turn/300,
                        C.ships.all, C.ships.my.all, C.ships.my.undocked, C.ships.my.docked,
                        C.ships.their.all, C.ships.their.docked, C.ships.their.undocked,
                        C.planets.all, C.planets.my.all, C.planets.my.dockable, C.planets.fresh, C.planets.their,
                        D.planets.all, D.planets.my, D.planets.their, D.planets.fresh, D.planets.dockable,
                        D.ships.all, D.ships.my, D.ships.their
                        ]

        #if np.mod(turn,6)==0:
            #output_vector = [random.randrange(-1000,1000)/1000.0, random.randrange(-1000,1000)/1000.0]
        #output_vector = [0.0,0.0]
        #logging.info('about to predict')
        output_vector = model.predict(np.array([input_vector]))[0]
        #logging.info(output_vector)
        W = Struct(
            threat=output_vector[0],  # attack/farm bias              +- 1
            aggro=output_vector[1],  # active/passive bias            +- 1
        )
        W.fill = 1.7 - W.aggro  # /\fill  \/acquire      ~0.2<2.2
        W.go_big = 0.7 + W.aggro  # /\go big  \/go close
        W.bastard = 0.4 + (W.aggro + W.threat) * 0.3  # /\attack docked  0.0<1.0
        W.defend = 1 - (W.aggro**2 - W.threat*0.5) * 0.5  # /\defend planet 0.0<1.0
        W.rank_bal = 7.5  # /\ conservative
        W.kamikaze = 15.5 + 3*(W.aggro + W.threat*1.2)

        timeout_lim = 1.975 - C.ships.all/1200
        command_queue = []

        # logging.info('ship start={}'.format(time.time() - start))
        for ship in S.my.undocked:
            #logging.info('ship {}'.format(ship.id))
            rank, target, distance = -2.2e-308, None, 0
            navigate_command = None

            if time.time() - start > timeout_lim:
                logging.info('break={}'.format(time.time() - start))
                break

            # For each planet in the game (only non-destroyed planets are included)
            # https://pythonprogramming.net/custom-ai-halite-ii-artificial-intelligence-competition/?completed=/modify-starter-bot-halite-ii-artificial-intelligence-competition/
            closest_empty_planets = game_map.nearby_entities_by_distance(ship, P.dockable)[:,1]
            closest_defend_planets_d = game_map.nearby_entities_by_distance(ship, P.my.all)[:,1]
            closest_defend_planets = [p for d, Planets in closest_defend_planets_d for p in Planets]
            closest_enemy_planets_d = game_map.nearby_entities_by_distance(ship, P.their)[:, 1]
            closest_enemy_ships_d = game_map.nearby_entities_by_distance(ship, S.their.all)[:10, 1]
            closest_enemy_ships = [s for d, Ships in closest_enemy_ships_d for s in Ships]
            closest_friendly_d = game_map.nearby_entities_by_distance(ship, S.my.all)[:5,1]
            closest_friendly = [s for d, Ships in closest_friendly_d for s in Ships]

            for dist, planets in closest_empty_planets:
                for planet in planets:
                    dist = np.sqrt(dist)
                    test_rank = W.rank_bal * \
                                (1-planet.ratio_docked**W.fill) * \
                                (planet.radius * planet.ratio_health * W.go_big)\
                                /((dist-planet.radius)+7)
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    elif test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                    #logging.info('planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

            for dist, enemies in closest_enemy_ships_d:
                dist = np.sqrt(dist)
                for enemy in enemies:
                    test_rank = \
                        (1-int(bool(enemy.docking_status==enemy.DockingStatus.UNDOCKED))*W.bastard*0.8) * \
                        (300-enemy.health) / (dist+15)
                    if test_rank > rank:
                        rank = test_rank
                        target = enemy
                        distance = dist
                    elif test_rank == rank:
                        if enemy.id < target.id:
                            rank = test_rank
                            target = enemy
                            distance = dist
                    #logging.info('enemy {} rank {} dist {}'.format(enemy.id, test_rank, dist))

            for dist, planets in closest_enemy_planets_d:
                for planet in planets:
                    dist = np.sqrt(dist) - planet.radius
                    rad = planet.radius
                    dock = planet.ratio_docked
                    health = planet.ratio_health
                    test_rank = W.bastard * W.kamikaze * rad * dock * (1-health*0.2) / ((dist-planet.radius) + 170)
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    elif test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                    #logging.info('enemy planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

            for dist, planets in closest_defend_planets_d:
                for planet in planets:
                    dist = np.sqrt(dist)
                    test_rank = W.defend * planet.ratio_docked * planet.radius * planet.ratio_health /((dist-planet.radius)+20)
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    elif test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                    #logging.info('defend planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

            if target:
                target_set = False
                if isinstance(target, hlt.entity.Planet):
                    if target.owner is None or (target.owner is not None and target.owner.id == me.id):
                        if target.ratio_docked < 1.0:
                            #logging.info('chose mining {}'.format(target.id))
                            check = MAX_SPEED + target.radius + DOCK_RADIUS

                            if ship.can_dock(target):
                                command_queue.append(ship.dock(target))
                            else:
                                speed = int(distance - target.radius - DOCK_RADIUS + 1) if distance <= check else MAX_SPEED
                                navigate_command = ship.navigate(
                                    target, game_map, closest_friendly + P.all + closest_enemy_ships,
                                    speed=speed, angular_step=2
                                )
                            target_set = True
                        else:  # defend
                            #logging.info('chose defending {}'.format(target.id))
                            if distance > target.radius + 8:
                                #logging.info('too far away {}'.format(distance))
                                navigate_command = ship.navigate(
                                    target, game_map, closest_friendly + P.all,
                                    speed=MAX_SPEED, angular_step=2
                                )
                            else:
                                angle = ship.calculate_angle_between(target)
                                # logging.info('angle {}'.format(angle))
                                angle = int((angle+90)%360)
                                # logging.info('angle {}'.format(angle))
                                new_target = hlt.entity.Position(
                                    COS_LOOKUP[angle] + ship.x,
                                    SIN_LOOKUP[angle] + ship.y,
                                )
                                # logging.info('new_target {}'.format(new_target))
                                navigate_command = ship.navigate(
                                    new_target, game_map, closest_friendly+closest_defend_planets,
                                    speed=5, angular_step=2
                                )
                                # logging.info('nav command set')
                            target_set = True
                if not target_set:
                    #if isinstance(target, hlt.entity.Ship):
                        #logging.info('chose hunting {}'.format(target.id))
                    #else:
                        #logging.info('chose kamikaze {}'.format(target.id))
                    navigate_command = ship.navigate(
                        target, game_map, P.all + closest_friendly + closest_enemy_ships, speed=MAX_SPEED)
                    target_set = True

            if navigate_command:
                command_queue.append(navigate_command)

        # if len(command_queue)==0:
        #     if len(my_docked)>0:
        #         navigate_command = my_docked[0].thrust(0,0)
        #     else:
        #         navigate_command = my_ships[0].thrust(0,0)
        #     command_queue.append(navigate_command)
        
        game.send_command_queue(command_queue)
        
        with open("c{}_input.vec".format(VERSION), "a") as f:
            f.write(str(input_vector))
            f.write('\n')

        with open("c{}_out.vec".format(VERSION), "a") as f:
            f.write(str(output_vector))
            f.write('\n')

        # logging.info('Finished turn {}'.format(turn))
        turn += 1

except ValueError:
    pass
except Exception as E:
    logging.exception('')


# GAME END
