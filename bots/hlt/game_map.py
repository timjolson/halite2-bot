from . import collision, entity
import numpy as np
import logging, time
from collections import OrderedDict
from .utils import Struct

class Map:
    """
    Map which houses the current game information/metadata.
    
    :ivar my_id: Current player id associated with the map
    :ivar width: Map width
    :ivar height: Map height
    """

    def __init__(self, my_id, width, height):
        """
        :param my_id: User's id (tag)
        :param width: Map width
        :param height: Map height
        """
        self.my_id = my_id
        self.width = width
        self.height = height
        self.area = width*height
        self._players = {}
        self._planets = {}

    def get_me(self):
        """
        :return: The user's player
        :rtype: Player
        """
        return self._players.get(self.my_id)

    def get_player(self, player_id):
        """
        :param int player_id: The id of the desired player
        :return: The player associated with player_id
        :rtype: Player
        """
        return self._players.get(player_id)

    def all_players(self):
        """
        :return: List of all players
        :rtype: list[Player]
        """
        return list(self._players.values())

    def get_planet(self, planet_id):
        """
        :param int planet_id:
        :return: The planet associated with planet_id
        :rtype: entity.Planet
        """
        return self._planets.get(planet_id)

    def all_planets(self):
        """
        :return: List of all planets
        :rtype: list[entity.Planet]
        """
        return list(self._planets.values())

    def nearby_entities_by_distance(self, entity, searchspace):
        """
        :param entity: The source entity to find distances from
        :return: Dict containing all entities with their designated distances
        :rtype: dict
        """
        result = {0:(-2.2e300,[])}
        i = 1
        # for foreign_entity in self._all_ships() + self.all_planets():
        for foreign_entity in searchspace:
            if entity == foreign_entity:
                continue
            result.setdefault(i, (entity.calculate_distance_between(foreign_entity), []))[1].append(foreign_entity)
            i+=1
        # logging.info('result1')
        # logging.info(result)
        result = np.array(sorted(result.items(), key=lambda t: t[1][0]), dtype=object)[1:]
        # logging.info('result2')
        # logging.info(result)
        return result

    def _link(self):
        """
        Updates all the entities with the correct ship and planet objects

        :return:
        """
        for celestial_object in self.all_planets() + self._all_ships():
            celestial_object._link(self._players, self._planets)

    def _parse(self, map_string):
        """
        Parse the map description from the game.

        :param map_string: The string which the Halite engine outputs
        :return: nothing
        """
        tokens = map_string.split()

        self._players, tokens = Player._parse(tokens)
        self._planets, tokens = entity.Planet._parse(tokens)

        assert(len(tokens) == 0)  # There should be no remaining tokens at this point
        self._link()

    def _all_ships(self):
        """
        Helper function to extract all ships from all players

        :return: List of ships
        :rtype: List[Ship]
        """
        all_ships = []
        for player in self.all_players():
            all_ships.extend(player.all_ships())
        return all_ships

    def their_ships(self):
        all_ships = []
        for player in [p for p in self.all_players() if p.id != self.my_id]:
            all_ships.extend(player.all_ships())
        return all_ships

    def _intersects_entity(self, target):
        """
        Check if the specified entity (x, y, r) intersects any planets. Entity is assumed to not be a planet.

        :param entity.Entity target: The entity to check intersections with.
        :return: The colliding entity if so, else None.
        :rtype: entity.Entity
        """
        for celestial_object in self._all_ships() + self.all_planets():
            if celestial_object is target:
                continue
            d = celestial_object.calculate_distance_between(target)
            if d <= (celestial_object.radius + target.radius) ** 2 + 0.05:
                return celestial_object
        return None

    def obstacles_between(self, ship, target, searchspace):
        """
        Check whether there is a straight-line path to the given point, without planetary obstacles in between.

        :param entity.Ship ship: Source entity
        :param entity.Entity target: Target entity
        :param entity.Entity ignore: Which entity type to ignore
        :return: The list of obstacles between the ship and target
        :rtype: list[entity.Entity]
        """
        obstacles = []
        # entities = ([] if issubclass(entity.Planet, ignore) else self.all_planets()) \
        #     + ([] if issubclass(entity.Ship, ignore) else self._all_ships())
        for foreign_entity in searchspace:
            # logging.info(foreign_entity)
            if foreign_entity == ship or foreign_entity == target:
                continue
            if collision.intersect_segment_circle(ship, target, foreign_entity, fudge=ship.radius + 0.3):
                obstacles.append(foreign_entity)
        return obstacles
    
    def sort_entities(game_map):
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
        P.my.all = [p for p in P.all if p.owner is not None and p.owner.id == game_map.my_id]
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
        C.planets.my.all = len(P.my.all)
        C.planets.my.dockable = len(P.my.dockable)
        C.planets.fresh = len(P.fresh)
        C.planets.their = len(P.their)
        C.planets.dockable = len(P.dockable)

        D = Struct(
            planets = Struct(
                all = 20*np.sum([p.radius for p in P.all])/game_map.area,
                my = 20*np.sum([p.radius for p in P.my.all])/game_map.area,
                their = 20*np.sum([p.radius for p in P.their])/game_map.area,
                fresh = 20*np.sum([p.radius for p in P.fresh])/game_map.area,
                dockable = 20*np.sum([p.radius for p in P.dockable])/game_map.area
            ),
            ships = Struct(
                all = 1000*C.ships.all/game_map.area,
                my = 1000*C.ships.my.all/game_map.area,
                their = 1000*C.ships.their.all/game_map.area
            )
        )
        
        all = game_map.all_players()
        num = len(all)
        me = game_map._players.get(game_map.my_id)
        all.remove(me)
        e1 = game_map._players.get(all[0].id).all_ships()
        e1 = [s for s in e1 if s.docking_status==s.DockingStatus.UNDOCKED]
        e2, e3 = None, None
        if num > 2:
            e2 = game_map._players.get(all[1].id).all_ships()
            e2 = [s for s in e2 if s.docking_status==s.DockingStatus.UNDOCKED]
            e3 = game_map._players.get(all[2].id).all_ships()
            e3 = [s for s in e3 if s.docking_status==s.DockingStatus.UNDOCKED]
        
        return S, P, C, D, e1, e2, e3

class Player:
    """
    :ivar id: The player's unique id
    """
    def __init__(self, player_id, ships={}):
        """
        :param player_id: User's id
        :param ships: Ships user controls (optional)
        """
        self.id = player_id
        self._ships = ships

    def all_ships(self):
        """
        :return: A list of all ships which belong to the user
        :rtype: list[entity.Ship]
        """
        return list(self._ships.values())

    def get_ship(self, ship_id):
        """
        :param int ship_id: The ship id of the desired ship.
        :return: The ship designated by ship_id belonging to this user.
        :rtype: entity.Ship
        """
        return self._ships.get(ship_id)

    @staticmethod
    def _parse_single(tokens):
        """
        Parse one user given an input string from the Halite engine.

        :param list[str] tokens: The input string as a list of str from the Halite engine.
        :return: The parsed player id, player object, and remaining tokens
        :rtype: (int, Player, list[str])
        """
        player_id, *remainder = tokens
        player_id = int(player_id)
        ships, remainder = entity.Ship._parse(player_id, remainder)
        player = Player(player_id, ships)
        return player_id, player, remainder

    @staticmethod
    def _parse(tokens):
        """
        Parse an entire user input string from the Halite engine for all users.

        :param list[str] tokens: The input string as a list of str from the Halite engine.
        :return: The parsed players in the form of player dict, and remaining tokens
        :rtype: (dict, list[str])
        """
        num_players, *remainder = tokens
        num_players = int(num_players)
        players = {}

        for _ in range(num_players):
            player, players[player], remainder = Player._parse_single(remainder)

        return players, remainder

    def __str__(self):
        return "Player {} with ships {}".format(self.id, self.all_ships())

    def __repr__(self):
        return self.__str__()

