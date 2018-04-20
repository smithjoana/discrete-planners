The ``discrete_planner`` module inside of this package contains useful features for running discrete planners.

In this module, I have written two discrete planner objects that can be instantiated for a holonomic robot that moves orthogonally in a static, 2D environment. In ``RandomPlanner``, the robot moves randomly through the environment until it either finds a path to the goal pose or the maximum number of steps is reached. In ``OptimalPlanner``, an A* heuristic search finds the shortest path (if it exists) between the robot's pose and the goal pose.

Check out the Wiki for more information!
