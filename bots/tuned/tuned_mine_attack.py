# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
import logging
import numpy as np
from hlt.utils import Struct #, imsave, flatten, UPLOADED
from collections import OrderedDict
from hlt.constants import *
import time

# GAME START
# Here we define the bot's name as Settler and initialize the game, including communication with the Halite engine.
game = hlt.Game("FuckResources", logging.WARNING)

try:
    W = Struct(

    )
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
            # ran_once = True

        # logging.info('filter start={}'.format(time.time() - start))
        # logging.info('Start turn {}'.format(turn))
        S = Struct(all=Struct(),my=Struct(),their=Struct())
        S.my.all = game_map.get_me().all_ships()
        S.my.docked = [s for s in S.my.all if s.DockingStatus == s.DockingStatus.DOCKED]
        S.my.undocked = [s for s in S.my.all if s not in S.my.docked]
        S.their.all = game_map.their_ships()
        S.their.docked = [s for s in S.their.all if s.DockingStatus == s.DockingStatus.DOCKED]
        S.their.undocked = [s for s in S.their.all if s not in S.their.docked]
        S.all = S.my.all + S.their.all

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
        C.ships.my.all = len(S.my.all)
        C.ships.my.docked = len(S.my.docked)
        C.ships.my.undocked = len(S.my.undocked)
        C.ships.their.all = len(S.their.all)
        C.ships.their.docked = len(S.their.docked)
        C.ships.their.undocked = len(S.their.undocked)
        C.planets.all = len(P.all)
        C.planets.my.all = len(P.my.all)
        C.planets.my.dockable = len(P.my.dockable)
        C.planets.fresh = len(P.fresh)
        C.planets.their = len(P.their)
        C.planets.dockable = len(P.dockable)

        if not ran_once:
            area = game.map.width * game.map.height
            DENSITIES = Struct(
                planets = Struct(
                    all = C.planets.all/area,
                    my = C.planets.my.all/area,
                    their = C.planets.their/area,
                    fresh = C.planets.fresh/area,
                    dockable = C.planets.dockable/area
                ),
                ships = Struct(
                    all = C.ships.all/area,
                    my = C.ships.my.all/area,
                    their = C.ships.their.all/area
                )
            )
        # logging.info('filter end={}'.format(time.time() - start))

        timeout_lim = 1.975 - C.ships.all/500

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
                logging.info('break={}'.format(time.time() - start))
                break

            # logging.info('distance start={}'.format(time.time() - start))
            # For each planet in the game (only non-destroyed planets are included)
            # https://pythonprogramming.net/custom-ai-halite-ii-artificial-intelligence-competition/?completed=/modify-starter-bot-halite-ii-artificial-intelligence-competition/
            closest_empty_planets = game_map.nearby_entities_by_distance(ship, P.dockable)[:,1]
            closest_enemy_ships_d = game_map.nearby_entities_by_distance(ship, S.their.all)[:10, 1]
            closest_enemy_ships = [s for d, Ships in closest_enemy_ships_d for s in Ships]
            closest_friendly_d = game_map.nearby_entities_by_distance(ship, S.my.all)[:5,1]
            closest_friendly = [s for d, Ships in closest_friendly_d for s in Ships]

            # logging.info('ship {}'.format(ship.id))
            rank, target, distance = -2.2e-308, None, 0
            for dist, planets in closest_empty_planets[:int(0.3*C.planets.dockable)]:
                for planet in planets:
                    # 1-2.5
                    # .5-2.5
                    # ~1
                    # 1-20
                    dist = np.sqrt(dist)
                    test_rank = (1-planet.ratio_docked**1.5) * (planet.radius*planet.ratio_health*1.0)/(dist+8)
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
                check = MAX_SPEED + target.radius + DOCK_RADIUS

                if ship.can_dock(target):
                    command_queue.append(ship.dock(target))
                else:
                    speed = int(distance - target.radius - DOCK_RADIUS + 1) if distance <= check else int(MAX_SPEED)
                    navigate_command = ship.navigate(
                        # ship.closest_point_to(planet),
                        target,
                        game_map,
                        closest_friendly+P.all,
                        speed=speed,
                        angular_step=2)

                    if navigate_command:
                        command_queue.append(navigate_command)
            else:  # no empty planets
                # logging.info('ship {}'.format(ship.id))
                rank, target = -2.2e308, None
                for dist, enemies in closest_enemy_ships_d:
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


        # if len(command_queue)==0:
        #     if len(my_docked)>0:
        #         navigate_command = my_docked[0].thrust(0,0)
        #     else:
        #         navigate_command = my_ships[0].thrust(0,0)
        #     command_queue.append(navigate_command)
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
