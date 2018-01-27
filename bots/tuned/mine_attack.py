# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
import logging
import numpy as np
from collections import OrderedDict
from hlt.constants import *
import time

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("MineFirstAskQuestionsLater", logging.WARNING)

me, turn, ran_once = None, 0, False

try:
    while True:
        # Update the map for the new turn and get the latest version
        game_map = game.update_map()
        start = time.time()

        if not ran_once:
            me = game_map.get_me()
            logging.info("Starting, my id: %d"%(me.id))
            ran_once = True

        # logging.info('Start turn {}'.format(turn))
        my_ships = game_map.get_me().all_ships()
        my_docked = [s for s in my_ships if s.DockingStatus == s.DockingStatus.DOCKED]
        their_ships = game_map.their_ships()
        all_planets = game_map.all_planets()
        my_planets = [p for p in all_planets if p.owner is not None and p.owner.id == me.id]
        fresh_planets = [p for p in all_planets if p.owner is None]
        their_planets = [p for p in all_planets if p not in my_planets+fresh_planets]
        dockable_planets = [p for p in fresh_planets+my_planets if p.ratio_docked<0.95]

        # logging.info('ships:{}vs{} ; planets owned:{}vs{} ; dockable:{}'.format(
        #     len(my_ships), len(their_ships), len(my_planets), len(their_planets), len(dockable_planets)
        #  ))

        # Here we define the set of commands to be sent to the Halite engine at the end of the turn
        command_queue = []
        # For every ship that I control
        for ship in my_ships:
            # If the ship is docked
            if ship.docking_status != ship.DockingStatus.UNDOCKED:
                # Skip this ship
                continue

            if time.time() - start > 1.6:
                break

            # For each planet in the game (only non-destroyed planets are included)
            # https://pythonprogramming.net/custom-ai-halite-ii-artificial-intelligence-competition/?completed=/modify-starter-bot-halite-ii-artificial-intelligence-competition/
            entities_by_distance = OrderedDict(sorted(game_map.nearby_entities_by_distance(ship).items(), key=lambda t: t[0]))
            closest_empty_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if
                                     isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and
                                     entities_by_distance[distance][0].ratio_docked < 0.95]
            empty_planet_distances = [distance for distance in entities_by_distance.keys() if
                                      isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and
                                      entities_by_distance[distance][0].ratio_docked < 0.95]
            closest_enemy_ships = [entities_by_distance[distance][0] for distance in entities_by_distance if
                                   isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and
                                   entities_by_distance[distance][0] not in my_ships]
            enemy_ship_distances = [distance for distance in entities_by_distance.keys() if
                                    isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and
                                    entities_by_distance[distance][0] in their_ships]

            if len(closest_empty_planets)>0:
                planet = closest_empty_planets[0]
                distance = np.sqrt(empty_planet_distances[0])
                check = MAX_SPEED
                check+=planet.radius
                check+=DOCK_RADIUS

                if ship.can_dock(planet):
                    command_queue.append(ship.dock(planet))
                else:
                    speed = int(distance - planet.radius - DOCK_RADIUS + 1) if distance <= check else int(MAX_SPEED)
                    navigate_command = ship.navigate(
                        # ship.closest_point_to(planet),
                        # hlt.entity.Position(planet.x, planet.y),
                        planet,
                        game_map,
                        speed=speed,
                        ignore_ships=False)

                    # logging.info('d{}, s{}, check{}'.format(distance, speed, (MAX_SPEED+planet.radius+DOCK_RADIUS)**2))

                    if navigate_command:
                        command_queue.append(navigate_command)

            else:  # no empty planets
                enemy = closest_enemy_ships[0]
                navigate_command = ship.navigate(
                    enemy, game_map, speed=MAX_SPEED, ignore_ships=True)
                if navigate_command:
                    command_queue.append(navigate_command)

        # Send our set of commands to the Halite engine for this turn
        if len(command_queue)==0:
            if len(my_docked)>0:
                navigate_command = my_docked[0].thrust(0,0)
            else:
                navigate_command = my_ships[0].thrust(0,0)
            command_queue.append(navigate_command)
        game.send_command_queue(command_queue)

        #  logging.info('Finished turn {}'.format(turn))
        turn += 1

except ValueError:
    pass
except:
    logging.Exception('')
    raise()

# GAME END
