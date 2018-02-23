# halite2-bot

Bots and related utilities for the Halite2 (2017-2018) AI/ML Competition. [My competition profile.](https://halite.io/user/?user_id=7900)

# My approach
The AI bot uses the big-picture status of a match to control weights for action prioritizing. Each ship then uses the weights and it's local environment to select an action.

Weights used (outputs from Keras NN)
    
    fill: prefer filling planets over acquiring
    attack: prefer attacking enemies over mining
    big: prefer big planets over nearer planets
    bastard: prefer attacking defenseless ships (docked)
    defend: use defensive mechanic (ship will orbit my planet)
    kamikaze: attack enemy planets instead of ships

My AI implementation is in /bots/MyBot.py. The AI script pulls dynamically from a pool of available NN models, of which there were initially many, some were removed due to ineffectiveness. Manually adjusted and randomly controlled bots are also in /bots.

## My Best Rank: Top 600
My best uploaded bot reached top 600 players with little training and without some gameplay mechanics.

<img width="300" src="/best.png">

****
## Added Mechanics & Improvements to Starterbot
Most of the improvements I've listed here take place within the hlt package.

### Obstacle checking
Made hlt.Map.obstacles_between() take a searchspace (list) of entities to check, rather than the entire map of objects. This is called inside navigate() for path selection. Greatly reduced processing time.

### Navigation
Made htl.entity.Ship.navigate() take a searchspace for obstacles_between().
Starterbot searched CCW up to default 90 degrees for valid path, improved to search CW and CCW, selecting whichever solution had fewer angle corrections. Searchspace reduced processing time. Path-finding 2 directions prevented getting stuck, made ships able to go either direction around a planet, hunt enemy ships through wider range of obstacles/directions.

### Defense Mechanic
Ships choosing to defend a friendly planet will orbit several units above the surface. This creates a reactive shield around docked ships and important planets.

### Pre-processing
#### Per Ship
Made hlt.Map.nearby_entities_by_distance() accept searchspace (list) of entities to check, rather than entire map of objects. Greatly reduced processing time.

#### Per Turn
Created hlt.utils.get_centroid() to calculate centroid of passed entities as percentage of map width/height.
These parameters are passed into the Keras NN to account for big-picture location of both my and enemy ships.

Created hlt.Map.sort_entities() to sort all game objects by type, docking status, owner, etc.
Returns nested object groups, sizes of the groups, area densities of planets and ships.
Many of these are passed into the Keras NN.

Examples:
    
    S  # Ship container
    S.my  # my ships container
    S.my.all  # all my ships
    S.my.undocked  # all my flying ships
    S.all  # All ships on map
    
    P  # Planet container
    P.all  # all planets on map
    P.dockable  # planets I can dock to
    
    C  # Counts container
    C.ships.all  # count of all ships on map
    C.ships.my.docked  # count of my docked ships
    C.planets.their  # count of enemy-owned planets
    
    D  # Densities container
    D.planets.my  # area density of planets I own
    D.ships.their  # area density of enemy ships
    
    e1, e2, e3  # Undocked enemy ship containers
