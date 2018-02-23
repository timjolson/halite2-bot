# Halite Starter Kit with my modifications
import hlt
# utils
import logging, time, os, random, subprocess, sys
import numpy as np

# Struct is a convenience class, get_centroid calculates area centroid of entities
from hlt.utils import Struct, get_centroid
# Halite game constants
from hlt.constants import *

# Disable stderr to import Keras and Tensorflow without breaking game server
stderr = sys.stderr  # restore this later
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'

# Keras, TF
import keras
import tensorflow as tf
from keras.models import load_model

# TF verbosity
tf.logging.set_verbosity(tf.logging.ERROR)

# Adjust config settings (can remove this for running on game server)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # % of GPU per session
set_session(tf.Session(config=config))

# restore stderr
sys.stderr = stderr

# Whether the bot is uploaded to game server or running locally
UPLOADED = not os.path.isfile('clear.bat')

# Get VERSION from filename
# VERSION = int(''.join(ele for ele in os.path.basename(__file__) if ele.isdigit()))
# Get VERSION from sys.argv
VERSION = eval(sys.argv[1])

# Read available model files
models = os.listdir('./models')
# Load the model for this bot version
model = load_model('./models/'+models[VERSION])

# Build lookup tables for sin/cos
COS_LOOKUP = {a:8*np.cos(np.deg2rad(a)) for a in np.arange(0,360,1)}
SIN_LOOKUP = {a:8*np.sin(np.deg2rad(a)) for a in np.arange(0,360,1)}

####
# GAME START
# Define the bot's name and initialize the game, including 
# communication with the Halite engine.
# Set logging level for local hlt.networking.Game
game = hlt.Game("v{}".format(VERSION), logging.INFO)

# Put entire loop inside try/except so we can 
# log errors (tracebacks from game are limited)
try:
    # Initial values, flag for having completed first turn
    me, turn, ran_once = None, 0, False
    
    # output_vector will be the output from the Keras NN
    # Each parameter adjusts how every ship handles game mechanics
    
    # Output vector for first turn
    output_vector = np.array([1.4,    1.0,   1.0,    .58,      1.0,     1.0])
    #                      fill   attack  big  bastard  defend  kamikaze
    # fill: prefer filling planets over acquiring
    # attack: prefer attacking enemies over mining
    # big: prefer big planets over nearer planets
    # bastard: prefer attacking defenseless ships (docked)
    # defend: use defensive mechanic (ship will orbit my planet)
    # kamikaze: attack enemy planets over their ships
    
    # Until the game server breaks this loop
    while True:
        # Update the map for the new turn and get the latest version
        game_map = game.update_map()
        # Turn start time (used for preventing time-outs (2 seconds for game server)
        start = time.time()
        
        # First turn
        if not ran_once:
            # My player on server
            me = game_map.get_me()
            # Number of players in game
            num_players = len(game_map.all_players())
            # Map Area
            area = game.map.width * game.map.height
            
            # If running locally, log
            if not UPLOADED:
                logging.info("Starting, my id: %d"%(me.id))
            
            # First turn completed
            ran_once = True

        
        ####
        # Start pre-processing for this turn
        
        logging.debug('Start turn {}'.format(turn))
        # Sort all game entities
        # S: ships
        # P: planets
        # C: count of ships/planets
        # D: density of ships/planets
        # e1-3: enemy ships that are not docked (by enemy)
        S, P, C, D, e1, e2, e3 = game_map.sort_entities()
        
        # Calculate centroids (0<centX<1, 0<centY<1)
        # My flying ships
        centroid_me = get_centroid(S.my.undocked, game.map.width, game.map.height)
        # Enemy N's flying ships
        centroid_e1 = get_centroid(e1, game.map.width, game.map.height)
        centroid_e2 = get_centroid(e2 if e2 else None, game.map.width, game.map.height)
        centroid_e3 = get_centroid(e3 if e3 else None, game.map.width, game.map.height)
        
        # Track time to make it this far
        logging.debug('filter end={}'.format(time.time() - start))
        
        # Change timeout cutoff as function of how many ships there are
        timeout_lim = 1.9 - C.ships.my.all/1000
        
        # Build input_vector to pass into Keras NN
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
        
        # Get output_vector from Keras NN
        new_output_vector = np.array(model.predict(np.array([input_vector]))[0])
        # Average the vector with the previous one to help smooth decisions
        output_vector = new_output_vector * 0.5 + output_vector * 0.5
        
        # Game mechanism weights from output_vector
        W = Struct(
            fill=output_vector[0],
            attack=output_vector[1],
            go_big=output_vector[2],
            bastard=output_vector[3],
            defend=output_vector[4],
            kamikaze=output_vector[5],
            rank_bal = 1.0
        )
        
        # These were some manually created functions for weights before using NN
        #~ W.fill = 1.7 - W.aggro  # /\fill  \/acquire      ~0.2<2.2
        #~ W.go_big = 0.7 + W.aggro  # /\go big  \/go close
        #~ W.bastard = 0.4 + (W.aggro + W.threat) * 0.3  # /\attack docked  0.0<1.0
        #~ W.defend = 1 - (W.aggro**2 - W.threat*0.5) * 0.5  # /\defend planet 0.0<1.0
        #~ W.kamikaze = 15.5 + 3*(W.aggro + W.threat*1.2)
        
        # List of commands to send to game server
        command_queue = []
        
        # end pre-processing and NN
        ####
        
        ####
        # Start looping through ships
        logging.debug('ship start={}'.format(time.time() - start))
        
        # For each of my ships not docked
        for ship in S.my.undocked:
            # Log id
            logging.debug('ship {}'.format(ship.id))
            
            ####
            # Initial values
            # rank: relative desire to perform an action
            # target: target of selected action
            # distance: distance to target
            rank, target, distance = -2.2e-308, None, 0
            
            # navigate_command will store a ship's thrust command if valid
            navigate_command = None

            # Check time-out to see if we need to stop analyzing ships
            elapsed = time.time() - start
            if elapsed > timeout_lim:
                logging.warning('break={}'.format(elapsed))
                break
            
            
            ####
            # Get nearby entities from provided entity search space
            
            # dockable planets
            closest_empty_planets = game_map.nearby_entities_by_distance(ship, P.dockable)[:,1]
            # planets to defend
            closest_defend_planets_d = game_map.nearby_entities_by_distance(ship, P.my.all)[:,1]
            closest_defend_planets = [p for d, Planets in closest_defend_planets_d for p in Planets]
            # enemy planets
            closest_enemy_planets_d = game_map.nearby_entities_by_distance(ship, P.their)[:, 1]
            # 10 enemy ships
            closest_enemy_ships_d = game_map.nearby_entities_by_distance(ship, S.their.all)[:10, 1]
            closest_enemy_ships = [s for d, Ships in closest_enemy_ships_d for s in Ships]
            # 5 friendly ships
            closest_friendly_d = game_map.nearby_entities_by_distance(ship, S.my.all)[:5,1]
            closest_friendly = [s for d, Ships in closest_friendly_d for s in Ships]
            
            # end getting neraby entities
            ####
            
            ####
            # Analyze action weights, store the best-so-far action, target, and distance
            
            # dockable planets to mine
            for dist, planets in closest_empty_planets:
                for planet in planets:
                    dist = np.sqrt(dist)
                    test_rank = W.rank_bal * \
                                (1-planet.ratio_docked**W.fill) * \
                                (planet.radius * planet.ratio_health * W.go_big)\
                                /((dist-planet.radius)+8)
                    
                    # store best-so-far
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    # edge case where 2 actions have same rank, select target by id
                    # rare cases caused jittering where target would alternate
                    elif test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                    
                    # Log action's weight and info
                    logging.debug('planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

            # enemies
            for dist, enemies in closest_enemy_ships_d:
                dist = np.sqrt(dist)
                for enemy in enemies:
                    test_rank = \
                        W.attack * \
                        (1-int(bool(enemy.docking_status==enemy.DockingStatus.UNDOCKED))*W.bastard*0.8) * \
                        (300-enemy.health) / (dist+300)
                    
                    # store best-so-far
                    if test_rank > rank:
                        rank = test_rank
                        target = enemy
                        distance = dist
                    # edge case where 2 actions have same rank, select target by id
                    # rare cases caused jittering where target would alternate
                    elif test_rank == rank:
                        if enemy.id < target.id:
                            rank = test_rank
                            target = enemy
                            distance = dist
                            
                    # Log action's weight and info
                    logging.debug('enemy {} rank {} dist {}'.format(enemy.id, test_rank, dist))
            
            # enemy planets
            for dist, planets in closest_enemy_planets_d:
                for planet in planets:
                    dist = np.sqrt(dist) - planet.radius
                    rad = planet.radius
                    dock = planet.ratio_docked
                    health = planet.ratio_health
                    
                    test_rank = W.bastard * W.kamikaze * rad * dock * (1-health*0.2) / ((dist-planet.radius) + 170)
                    
                    # store best-so-far
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    # edge case where 2 actions have same rank, select target by id
                    # rare cases caused jittering where target would alternate
                    elif test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                            
                    # Log action's weight and info
                    logging.debug('enemy planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

            # my planets to defend
            for dist, planets in closest_defend_planets_d:
                for planet in planets:
                    dist = np.sqrt(dist)
                    
                    test_rank = W.defend * planet.ratio_docked * planet.radius * planet.ratio_health /((dist-planet.radius)+20)
                    
                    # store best-so-far
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    # edge case where 2 actions have same rank, select target by id
                    # rare cases caused jittering where target would alternate
                    elif test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                            
                    # Log action's weight and info
                    logging.debug('defend planet {} rank {} dist {}'.format(planet.id, test_rank, dist))
            
            # end analyzing actions
            ####
            
            ####
            # Perform action based on ranking selection
            if target:
                
                target_set = False
                
                # If it's a planet
                if isinstance(target, hlt.entity.Planet):
                    # If it is unowned or it's mine
                    if target.owner is None or (target.owner is not None and target.owner.id == me.id):
                        # Ship chose to dock and mine the planet
                        if target.ratio_docked < 1.0:
                            logging.debug('chose mining {}'.format(target.id))
                            
                            # radius to slow down ship
                            check = MAX_SPEED + target.radius + DOCK_RADIUS
                            
                            # try to dock
                            if ship.can_dock(target):
                                command_queue.append(ship.dock(target))
                            else:
                                # can't dock, move to it, using lower speed when close
                                speed = int(distance - target.radius - DOCK_RADIUS + 1) if distance <= check else MAX_SPEED
                                
                                # create thrust command, avoid nearby ships and all planets
                                # 2 degree increment path planning
                                navigate_command = ship.navigate(
                                    target, game_map, closest_friendly + P.all + closest_enemy_ships,
                                    speed=speed, angular_step=2
                                )
                                
                            # we have selected a target for this ship
                            target_set = True
                        
                        # Ship chose to defend the planet
                        else:
                            logging.debug('chose defending {}'.format(target.id))
                            
                            # If outside orbit range, move toward planet
                            if distance > target.radius + 8:
                                logging.debug('too far away {}'.format(distance))
                                navigate_command = ship.navigate(
                                    target, game_map, closest_friendly + P.all,
                                    speed=MAX_SPEED, angular_step=2
                                )
                            # Within orbit range
                            else:
                                # Get angle to planet center
                                angle = ship.calculate_angle_between(target)
                                # Add 90 degrees counter-clockwise
                                angle = int((angle+90)%360)
                                
                                # Get orbit target position
                                new_target = hlt.entity.Position(
                                    COS_LOOKUP[angle] + ship.x,
                                    SIN_LOOKUP[angle] + ship.y,
                                )
                                
                                # Get thrust command, avoid friendly ships and planets
                                navigate_command = ship.navigate(
                                    new_target, game_map, closest_friendly+closest_defend_planets,
                                    speed=5, angular_step=2
                                )
                                
                            # we have selected a target for this ship
                            target_set = True
                            
                # We have not selected a friendly target planet for the ship
                if not target_set:
                    # Chose attack enemy ship
                    if isinstance(target, hlt.entity.Ship):
                        logging.debug('chose hunting {}'.format(target.id))
                    # Chose attack enemy planet
                    else:
                        logging.debug('chose kamikaze {}'.format(target.id))
                    
                    # Navigate to target, avoiding obstacles
                    navigate_command = ship.navigate(
                        target, game_map, P.all + closest_friendly + closest_enemy_ships, speed=MAX_SPEED)
                    
                    # we have selected a target for this ship
                    target_set = True
            
            # end performing action
            ####
            
            # If we've successfully created a thrust command for this ship
            if navigate_command:
                # Append to commands to send game server
                command_queue.append(navigate_command)
            
        # END FOR EACH SHIP
        ####
        
        ####
        # Send commands to game server
        
        # If there are no commands, send something so game server doesn't
        # time-out our bot
        if len(command_queue)==0:
            # send one to a docked ship
            if C.ships.my.docked>0:
                navigate_command = S.my.docked[0].thrust(0,0)
            # send one to lowest id ship
            else:
                navigate_command = S.my.all[0].thrust(0,0)
            
            # append command
            command_queue.append(navigate_command)
        
        # Transmit commands
        game.send_command_queue(command_queue)
        
        ####
        # Running locally, save input and output vectors to training data files
        if not UPLOADED:
            with open("{}_{}_input.vec".format(me.id, VERSION), "a") as f:
                f.write(str(input_vector))
                f.write('\n')

            with open("{}_{}_out.vec".format(me.id, VERSION), "a") as f:
                f.write(str(list(output_vector)))
                f.write('\n')
        
        # Done with this turn
        logging.debug('Finished turn {}'.format(turn))
        turn += 1

# A ValueError is typically raised if the bot is destroyed before the game ends
#except ValueError:
#    pass

# Log exceptions so we can actually see them
except Exception as E:
    logging.exception('')


# GAME END
