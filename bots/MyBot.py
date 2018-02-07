# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
import logging, time, os, random, subprocess, sys
import numpy as np
from hlt.utils import Struct, get_centroid
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

UPLOADED = not os.path.isfile('clear.bat')

# from file name
# VERSION = int(''.join(ele for ele in os.path.basename(__file__) if ele.isdigit()))
# from passed in argument
VERSION = eval(sys.argv[1])

models = os.listdir('./models')
model = load_model('./models/'+models[VERSION])

COS_LOOKUP = {a:8*np.cos(np.deg2rad(a)) for a in np.arange(0,360,1)}
SIN_LOOKUP = {a:8*np.sin(np.deg2rad(a)) for a in np.arange(0,360,1)}

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("v{}".format(VERSION), logging.INFO)

try:
    me, turn, ran_once = None, 0, False
    output_vector = np.array([1.4,    1.0,   1.0,    .58,      1.0,     1.0])
    #                      fill  attaack  big  bastard  defend  kamikaze
    
    while True:
        # Update the map for the new turn and get the latest version
        game_map = game.update_map()
        start = time.time()
        
        if not ran_once:
            me = game_map.get_me()
            num_players = len(game_map.all_players())
            area = game.map.width * game.map.height
            if not UPLOADED:
                logging.info("Starting, my id: %d"%(me.id))
            ran_once = True

        logging.debug('Start turn {}'.format(turn))
        S, P, C, D, e1, e2, e3 = game_map.sort_entities()
        centroid_me = get_centroid(S.my.undocked, game.map.width, game.map.height)
        centroid_e1 = get_centroid(e1, game.map.width, game.map.height)
        centroid_e2 = get_centroid(e2 if e2 else None, game.map.width, game.map.height)
        centroid_e3 = get_centroid(e3 if e3 else None, game.map.width, game.map.height)
        logging.debug('filter end={}'.format(time.time() - start))
        
        timeout_lim = 1.9 - C.ships.my.all/1000
        
        input_vector = [num_players, area/98304, turn/300,
                        C.ships.all, C.ships.my.all, C.ships.my.undocked, C.ships.my.docked,
                        C.ships.their.all, C.ships.their.docked, C.ships.their.undocked,
                        C.planets.all, C.planets.my.all, C.planets.my.dockable, C.planets.fresh, C.planets.their,
                        D.planets.all, D.planets.my, D.planets.their, D.planets.fresh, D.planets.dockable,
                        D.ships.all, D.ships.my, D.ships.their, 
                        centroid_me[0], centroid_me[1],
                        centroid_e1[0], centroid_e1[1],
                        centroid_e2[0], centroid_e2[1],
                        centroid_e3[0], centroid_e3[1],
                        ]

        new_output_vector = np.array(model.predict(np.array([input_vector]))[0])
        output_vector = new_output_vector * 0.5 + output_vector * 0.5
        
        logging.debug(output_vector)
        W = Struct(
            fill=output_vector[0],
            attack=output_vector[1],
            go_big=output_vector[2],
            bastard=output_vector[3],
            defend=output_vector[4],
            kamikaze=output_vector[5],
            rank_bal = 1.0
        )
        #~ W.fill = 1.7 - W.aggro  # /\fill  \/acquire      ~0.2<2.2
        #~ W.go_big = 0.7 + W.aggro  # /\go big  \/go close
        #~ W.bastard = 0.4 + (W.aggro + W.threat) * 0.3  # /\attack docked  0.0<1.0
        #~ W.defend = 1 - (W.aggro**2 - W.threat*0.5) * 0.5  # /\defend planet 0.0<1.0
        #~ W.kamikaze = 15.5 + 3*(W.aggro + W.threat*1.2)
        
        command_queue = []

        logging.debug('ship start={}'.format(time.time() - start))
        for ship in S.my.undocked:
            logging.debug('ship {}'.format(ship.id))
            rank, target, distance = -2.2e-308, None, 0
            navigate_command = None

            elapsed = time.time() - start
            if elapsed > timeout_lim:
                logging.warning('break={}'.format(elapsed))
                break

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
                                /((dist-planet.radius)+8)
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    elif test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                    logging.debug('planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

            for dist, enemies in closest_enemy_ships_d:
                dist = np.sqrt(dist)
                for enemy in enemies:
                    test_rank = \
                        W.attack * \
                        (1-int(bool(enemy.docking_status==enemy.DockingStatus.UNDOCKED))*W.bastard*0.8) * \
                        (300-enemy.health) / (dist+300)
                    if test_rank > rank:
                        rank = test_rank
                        target = enemy
                        distance = dist
                    elif test_rank == rank:
                        if enemy.id < target.id:
                            rank = test_rank
                            target = enemy
                            distance = dist
                    logging.debug('enemy {} rank {} dist {}'.format(enemy.id, test_rank, dist))

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
                    logging.debug('enemy planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

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
                    logging.debug('defend planet {} rank {} dist {}'.format(planet.id, test_rank, dist))
            
            if target:
                target_set = False
                if isinstance(target, hlt.entity.Planet):
                    if target.owner is None or (target.owner is not None and target.owner.id == me.id):
                        if target.ratio_docked < 1.0:
                            logging.debug('chose mining {}'.format(target.id))
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
                            logging.debug('chose defending {}'.format(target.id))
                            if distance > target.radius + 8:
                                logging.debug('too far away {}'.format(distance))
                                navigate_command = ship.navigate(
                                    target, game_map, closest_friendly + P.all,
                                    speed=MAX_SPEED, angular_step=2
                                )
                            else:
                                angle = ship.calculate_angle_between(target)
                                angle = int((angle+90)%360)
                                new_target = hlt.entity.Position(
                                    COS_LOOKUP[angle] + ship.x,
                                    SIN_LOOKUP[angle] + ship.y,
                                )
                                navigate_command = ship.navigate(
                                    new_target, game_map, closest_friendly+closest_defend_planets,
                                    speed=5, angular_step=2
                                )
                            target_set = True
                if not target_set:
                    if isinstance(target, hlt.entity.Ship):
                        logging.debug('chose hunting {}'.format(target.id))
                    else:
                        logging.debug('chose kamikaze {}'.format(target.id))
                    navigate_command = ship.navigate(
                        target, game_map, P.all + closest_friendly + closest_enemy_ships, speed=MAX_SPEED)
                    target_set = True

            if navigate_command:
                command_queue.append(navigate_command)

        if len(command_queue)==0:
            if C.ships.my.docked>0:
                navigate_command = S.my.docked[0].thrust(0,0)
            else:
                navigate_command = S.my.all[0].thrust(0,0)
            command_queue.append(navigate_command)
        
        game.send_command_queue(command_queue)
        
        if not UPLOADED:
            with open("{}_{}_input.vec".format(me.id, VERSION), "a") as f:
                f.write(str(input_vector))
                f.write('\n')

            with open("{}_{}_out.vec".format(me.id, VERSION), "a") as f:
                f.write(str(list(output_vector)))
                f.write('\n')
        
        logging.debug('Finished turn {}'.format(turn))
        turn += 1

#except ValueError:
#    pass
except Exception as E:
    logging.exception('')


# GAME END
