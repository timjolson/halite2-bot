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
game = hlt.Game("IAmTuned", logging.WARNING)

try:
    # if not UPLOADED():
    # from hlt.layers import Images
    # Layer = Images.Layer
    #
    # shape = [game.map.height * hlt.layers.PIXELS_PER_UNIT, game.map.width * hlt.layers.PIXELS_PER_UNIT]
    #
    # Images.load()
    #
    # def blank_map():
    #     ret = Layer(shape)
    #     ret.offset = 0
    #     return ret
    #
    # channels = Struct(
    #         my_ships=blank_map(),
    #         my_docked=blank_map(),
    #         my_ships_density=blank_map(),
    #         their_ships=blank_map(),
    #         their_docked=blank_map(),
    #         their_ships_density=blank_map(),
    #         my_planets=blank_map(),
    #         their_planets=blank_map(),
    #         fresh_planets=blank_map(),
    #         dockable_planets=blank_map()
    #     )

    me, turn, ran_once = None, 0, False

    while True:
        # Update the map for the new turn and get the latest version
        game_map = game.update_map()
        start = time.time()

        if not ran_once:
            me = game_map.get_me()
            # logging.info("Starting, my id: %d"%(me.id))
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

        timeout_lim = 1.975 - len(S.my.all)/800

        # if not UPLOADED():
        # if me.id == 0:
            # logging.info('loop start={}'.format(time.time() - start))
            # for p in my_planets:
            #     #  logging.debug('p{} dock:{}'.format(p.id, p.ratio_docked))
            #     if p.ratio_docked < 0.95:
            #         img = Images.Planet.dockable(p.radius, p.x, p.y)
            #         flatten(channels.dockable_planets, img * (1-p.ratio_docked)*255.0)
            #         flatten(channels.my_planets, img * p.ratio_health)
            #     else:
            #         img = Images.Planet.full(p.radius, p.x, p.y)
            #         flatten(channels.my_planets, img * p.ratio_health * 255.0)
            # for p in fresh_planets:
            #     img = Images.Planet.dockable(p.radius, p.x, p.y)
            #     flatten(channels.dockable_planets, img * p.ratio_health * 255.0)
            # for p in their_planets:
            #     img = Images.Planet.full(p.radius, p.x, p.y)
            #     flatten(channels.their_planets, img * p.ratio_health * 255.0)
            #
            # img = Images.Ship.friendly()
            # for s in my_ships:
            #     # logging.info('ship {} at {},{}'.format(s.id, s.x, s.y))
            #     img.pos(s.x, s.y)
            #     flatten(channels.my_ships, img * float(s.health))
            #
            # img = Images.Ship.armed()
            # for s in their_ships:
            #     img.pos(s.x, s.y)
            #     flatten(channels.their_ships, img * float(s.health))
            #
            # img_map = np.dstack([
            #     # channels.my_ships,
            #     # channels.my_planets,
            #     flatten(channels.my_planets, channels.my_ships),
            #     channels.dockable_planets,
            #     # channels.their_ships,
            #     # channels.their_planets,
            #     flatten(channels.their_ships, channels.their_planets),
            # ])
            # # cv2.imshow('', np.uint8(channels.dockable_planets*255))
            # # cv2.waitKey(0)
            # imsave('frames\img%d.bmp' % (turn), img_map)

        # Here we define the set of commands to be sent to the Halite engine at the end of the turn
        command_queue = []
        # For every ship that I control
        # logging.info('ship start={}'.format(time.time() - start))
        for ship in S.my.undocked:
            if time.time() - start > timeout_lim:
                # logging.info('break={}'.format(time.time() - start))
                break

            # logging.info('distance start={}'.format(time.time() - start))
            # For each planet in the game (only non-destroyed planets are included)
            # https://pythonprogramming.net/custom-ai-halite-ii-artificial-intelligence-competition/?completed=/modify-starter-bot-halite-ii-artificial-intelligence-competition/
            # entities_by_distance = OrderedDict(sorted(game_map.nearby_entities_by_distance(ship).items(), key=lambda t: t[0]))
            closest_empty_planets = game_map.nearby_entities_by_distance(ship, P.dockable)[:,1]
            closest_enemy_ships = game_map.nearby_entities_by_distance(ship, S.their.all)[:10, 1]

            closest_friendly = game_map.nearby_entities_by_distance(ship, S.my.all)[:5,1]
            closest_friendly = [s for d, S in closest_friendly for s in S]

            # logging.info('ship {}'.format(ship.id))
            rank, target, distance = -2.2e-308, None, 0
            # logging.info('closest')
            # logging.info([p.id for d, P in closest_empty_planets for p in P])
            obstacles = [ship for d, Ships in closest_enemy_ships for ship in Ships]
            for dist, planets in closest_empty_planets:
                # logging.debug('obstacles')
                # logging.debug(obstacles)
                # obstacles.extend(planets)

                for planet in planets:
                    # 1-2.5
                    # .5-2.5
                    # ~1
                    # 1-20
                    test_rank = (1-planet.ratio_docked**1.5) * (planet.radius*planet.ratio_health*1.0)/(np.sqrt(dist)+8)
                    if test_rank > rank:
                        rank = test_rank
                        target = planet
                        distance = dist
                    if test_rank == rank:
                        if planet.id < target.id:
                            rank = test_rank
                            target = planet
                            distance = dist
                    # logging.info('planet {} rank {} dist {}'.format(planet.id, test_rank, dist))

            if target:
                # logging.info('chose {}'.format(target.id))
                distance = np.sqrt(distance)
                check = MAX_SPEED
                check += target.radius
                check += DOCK_RADIUS

                if ship.can_dock(target):
                    command_queue.append(ship.dock(target))
                else:
                    speed = int(distance - target.radius - DOCK_RADIUS + 1) if distance <= check else int(MAX_SPEED)
                    navigate_command = ship.navigate(
                        # ship.closest_point_to(planet),
                        # hlt.entity.Position(planet.x, planet.y),
                        target,
                        game_map,
                        obstacles+closest_friendly+P.all,
                        speed=speed,
                        angular_step=1)

                    if navigate_command:
                        command_queue.append(navigate_command)
            else:  # no empty planets
                # logging.info('ship {}'.format(ship.id))
                rank, target = -2.2e308, None
                for dist, enemies in closest_enemy_ships:
                    for enemy in enemies:
                        test_rank = \
                            (1-int(bool(enemy.docking_status==enemy.DockingStatus.UNDOCKED))*0.8) * enemy.health / (np.sqrt(dist)+300)
                        if test_rank > rank:
                            rank = test_rank
                            target = enemy
                        # logging.info('ship {} rank {} dist {}'.format(enemy.id, test_rank, dist))
                if target:
                    # logging.info('chose hunting {}'.format(target.id))
                    navigate_command = ship.navigate(
                        target, game_map, P.all+closest_friendly, speed=MAX_SPEED)
                    if navigate_command:
                        command_queue.append(navigate_command)

        if len(command_queue)==0:
            if len(S.my.docked)>0:
                navigate_command = S.my.docked[0].thrust(0,0)
            else:
                navigate_command = S.my.all[0].thrust(0,0)
            command_queue.append(navigate_command)
        game.send_command_queue(command_queue)

        # if not UPLOADED():
        #     if me.id == 0:
        #         channels = Struct(
        #             my_ships=blank_map(),
        #             my_docked=blank_map(),
        #             my_ships_density=blank_map(),
        #             their_ships=blank_map(),
        #             their_docked=blank_map(),
        #             their_ships_density=blank_map(),
        #             my_planets=blank_map(),
        #             their_planets=blank_map(),
        #             fresh_planets=blank_map(),
        #             dockable_planets=blank_map()
        #         )

        # logging.info('Finished turn {}'.format(turn))
        turn += 1

#except ValueError:
#    pass
except Exception as E:
    logging.exception('')
    raise(E)

# GAME END
