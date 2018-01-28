# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
import logging
import numpy as np
from hlt.utils import Struct, imsave, flatten, UPLOADED
from collections import OrderedDict
from hlt.constants import *
import time

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("MineBot3000", logging.INFO)

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

        S = Struct(all=Struct(),my=Struct(),their=Struct())
        S.my.all = game_map.get_me().all_ships()
        S.my.docked = [s for s in S.my.all if s.docking_status != s.DockingStatus.UNDOCKED]
        S.my.undocked = [s for s in S.my.all if s.docking_status == s.DockingStatus.UNDOCKED]
        S.their.all = game_map.their_ships()
        S.their.docked = [s for s in S.their.all if s.docking_status != s.DockingStatus.UNDOCKED]
        S.their.undocked = [s for s in S.their.all if s.docking_status == s.DockingStatus.UNDOCKED]
        S.all = S.my.all + S.their.all

        P = Struct(all=Struct(), my=Struct(), their=Struct())
        P.all = game_map.all_planets()
        P.my.all = [p for p in P.all if p.owner is not None and p.owner.id == me.id]
        P.my.dockable = [p for p in P.my.all if p.ratio_docked < 1.0]
        P.fresh = [p for p in P.all if p.owner is None]
        P.dockable = [p for p in P.fresh+P.my.dockable]
        P.their = [p for p in P.all if p not in P.my.all+P.fresh]

        # Here we define the set of commands to be sent to the Halite engine at the end of the turn
        command_queue = []
        # For every ship that I control
        for ship in S.my.undocked:
            if time.time() - start > 1.6:
                break

            # For each planet in the game (only non-destroyed planets are included)
            # https://pythonprogramming.net/custom-ai-halite-ii-artificial-intelligence-competition/?completed=/modify-starter-bot-halite-ii-artificial-intelligence-competition/
            closest_empty_planets = game_map.nearby_entities_by_distance(ship, P.dockable)[:,1]
            closest_enemy_ships = game_map.nearby_entities_by_distance(ship, S.their.all)[:10, 1]
            
            obstacles = [ship for d, Ships in closest_enemy_ships for ship in Ships]
            if len(closest_empty_planets)>0:
                dist, planet = closest_empty_planets[0]
                planet = planet[0]
                distance = np.sqrt(dist)
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
                        obstacles+S.my.all+P.all,
                        speed=speed,
                        ignore_ships=False)

                    # logging.info('d{}, s{}, check{}'.format(distance, speed, (MAX_SPEED+planet.radius+DOCK_RADIUS)**2))

                    if navigate_command:
                        command_queue.append(navigate_command)

        # Send our set of commands to the Halite engine for this turn
        if len(command_queue)==0:
            if len(S.my.docked)>0:
                navigate_command = S.my.docked[0].thrust(0,0)
            else:
                navigate_command = S.my.all[0].thrust(0,0)
            command_queue.append(navigate_command)
        game.send_command_queue(command_queue)

        #  logging.info('Finished turn {}'.format(turn))
        turn += 1

#except ValueError:
#    pass
except:
    logging.exception('')
    raise()

# GAME END
