PIXELS_PER_UNIT = 3  # number of game units per pixel
UNITS_PER_PIXEL = 1 / PIXELS_PER_UNIT  # number of pixels per game unit

SHIP_RADIUS = 0.5  #: Radius of a ship
MAX_SHIP_HEALTH = 255  #: Starting health of ship, also its max
BASE_SHIP_HEALTH = 255  #: Starting health of ship, also its max
WEAPON_COOLDOWN = 1  #: Weapon cooldown period
WEAPON_RADIUS = 5.0  #: Weapon damage radius
WEAPON_DAMAGE = 64  #: Weapon damage
SHIP_SHAPE = [int(WEAPON_RADIUS * 5 * PIXELS_PER_UNIT)+1] * 2  # max shape of a ship
SHIP_CENTER = tuple([int(SHIP_SHAPE[0] / 2)] * 2)  # center of ship image
MAX_SPEED = 7  #: Max number of units of distance a ship can travel in a turn

EXPLOSION_RADIUS = 5.0  #: Radius in which explosions affect other entities
DOCK_RADIUS = 4.0  #: Distance from the edge of the planet at which ships can try to dock

#: Number of turns it takes to dock a ship
DOCK_TURNS = 5
#: Number of production units per turn contributed by each docked ship
BASE_PRODUCTIVITY = 6
#: Distance from the planets edge at which new ships are created
SPAWN_RADIUS = 2.0
