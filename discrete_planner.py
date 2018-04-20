import numpy as np
from random import randint
from math import isinf


class DiscretePlanner(object):
    """Discrete planners to be used for a holonomic robot in a static, 2D environment."""

    def set_attributes(self, world_state, robot_pose, goal_pose):
        """
        Initialize attributes. 

        :type world_state: 2D list
        :param world_state: represents the world state, where 0 indicates a navigable space and 1 indicates an occupied/obstacle space

        :type robot_pose: tuple
        :param robot_pose: represents the robot pose (x,y) in the world state

        :type goal_pose: tuple
        :param goal_pose: represents the goal pose (x,y) in the world state
        """
        self.ws = world_state
        self.rp = robot_pose
        self.gp = goal_pose
        self.x_lim = len(self.ws)
        self.y_lim = len(self.ws[0])
        self.ws_b = np.array(self.ws, dtype=bool)

    def get_orthogonal_points(self, point):
        """
        Generate list of 4 orthogonal points.
        
        :param point: a point [x,y] in the world state

        :return: 2D list of 4 orthogonal [x,y] points
        """
        orthogonal_points = [
            [point[0] + 1, point[1]],
            [point[0] - 1, point[1]],
            [point[0], point[1] + 1],
            [point[0], point[1] - 1],
        ]
        return orthogonal_points

    def check_boundary(self, points):
        """ 
        Remove points that are outside the world_state boundary. 

        :param points: 2D list of [x,y] points 

        :return: 2D list of [x,y] points inside the world_state boundary
        """
        for i, [x, y] in enumerate(points, start=0):
            if (0 <= x <= self.x_lim) and (0 <= y <= self.y_lim):
                pass
            else:
                points[i] = 0
        points[:] = (value for value in points if value != 0)
        return points

    def check_obstacles(self, points):
        """
        Remove points that are in an occupied/obstacle space.

        :param points: 2D list of [x,y] points inside the world_state boundary

        :return: a 2D list of [x,y] points that are additionally not in occupied/obstacle space
        """
        x_max = self.x_lim
        y_max = self.y_lim
        ws_b = self.ws_b

        # Check to see if each point could corresponding to the top left, top
        # right, bottom left, or bottom right corner of an obstacle.
        for i, [x, y] in enumerate(points, start=0):
            top_left = (x < x_max) and (y < y_max) and (ws_b[x][y])
            top_right = (x < x_max) and (0 < y <= y_max) and (ws_b[x][y - 1])
            bot_left = (0 < x <= x_max) and (y < y_max) and (ws_b[x - 1][y])
            bot_right = (0 < x <= x_max) and (0 < y <= y_max) and (
                ws_b[x - 1][y - 1])

            if top_left or top_right or bot_left or bot_right:
                points[i] = 0

        points[:] = (value for value in points if value != 0)
        return points

    def plot_path(self, path):
        """
        Plot the path if a path was found.

        :param path: None or a list of (x,y) tuples that describe a path between the robot_pose and the goal_pose
        """
        if path:
            # Plotting is an additional feature I have incorporated and was not
            # part of the instructions. I have chosen to import the matplotlib
            # library here so the user can still perform a path search with
            # only the numpy external library.
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, aspect='equal')
            ax1.set_xlim([-0.2, len(self.ws[0]) + 0.2])
            ax1.set_ylim([-0.2, len(self.ws) + 0.2])
            plt.gca().invert_yaxis()
            ax1.xaxis.tick_top()
            plt.grid()

            # Add obstacles, rp, gp, and path to the figure.
            obstacles = np.asarray(self.ws).T
            obstacles = list(zip(*np.where(obstacles)))  # [y,x] order
            for obs in obstacles:
                patch_obs = ax1.add_patch(
                    patches.Rectangle(obs, 1, 1, facecolor="black"),
                )

            point_rp = plt.scatter(
                self.rp[1], self.rp[0], 
                c='blue', edgecolor='none', s=100,
            )
            point_gp = plt.scatter(
                self.gp[1], self.gp[0],
                c='red', edgecolor='none', s=100,
            )

            x_val = [x[1] for x in path]
            y_val = [x[0] for x in path]
            line_path, = plt.plot(x_val, y_val, '--', c='red')

            # Create a legend.
            if obstacles:
                plt.legend(
                    [point_rp, point_gp, line_path, patch_obs],
                    ['Robot', 'Goal', 'Path', 'Obstacle'],
                    fontsize="small",
                )
            else:
                plt.legend(
                    [point_rp, point_gp, line_path],
                    ['Robot', 'Goal', 'Path'],
                    fontsize="small",
                )

            plt.xticks(np.arange(0, len(self.ws[0]) + 1, 1.0))
            plt.yticks(np.arange(0, len(self.ws) + 1, 1.0))
            plt.show()


class RandomPlanner(DiscretePlanner):
    """A random planner."""

    def set_max_step_number(self, max_step_number):
        """
        Initialize parameter.

        :type max_step_number: int >=0 
        :param max_step_number: the maximum number of steps the random planner can take to find the goal_pose.
        """
        self.msn = max_step_number

    def search(self, world_state, robot_pose, goal_pose):
        """
        Execute the random planner.

        :type world_state: 2D list
        :param world_state: represents the world state, where 0 indicates a navigable space and 1 indicates an occupied/obstacle space

        :type robot_pose: tuple
        :param robot_pose: represents the robot pose (x,y) in the world state

        :type goal_pose: tuple
        :param goal_pose: represents the goal pose (x,y) in the world state

        :return: None or a list of (x,y) tuples that describe a feasible path from the robot_pose to the goal_pose
        """
        self.set_attributes(world_state, robot_pose, goal_pose)

        step = 1
        cp = self.rp        # initialize current pose to be the robot pose
        path = [0 for value in range(self.msn + 1)]
        path[0] = cp

        # Continue to search for feasible random moves and add them to the
        # path.
        while (step <= self.msn) and (cp != self.gp):
            list0 = self.get_orthogonal_points(cp)
            list1 = self.check_boundary(list0)
            list2 = self.check_obstacles(list1)
            list3 = self.check_recent(list2[:], path[:])

            # As per the instructions, if all possible next moves have been
            # visited recently, allow planner to visit them again.
            if not list3:
                list3 = list2

            # We could find ourselves at a point that it is surrounded by the
            # boundary and/or obstacle space.
            if not list3:
                break

            next_pose = self.select_next_pose(list3)
            path[step] = next_pose
            cp = next_pose
            step += 1

        path = self.calculate_path(cp, path)
        return path

    def check_recent(self, points, recent):
        """
        Remove points from list that have been visited recently. As per the instructions, the random planner has a short memory and will never attempt to visit a cell that was visited in the last sqrt(max_step_number) steps, except if this is the only available option.

        :param points: a 2D list of [x,y] points
        :param recent: list of (x,y) tuples that describe the random path up to this point.

        :return: a 2D list of [x,y] points that have not been visited recently
        """
        num = int(self.msn**0.5)

        # recent may contain trailing 0's if this method was called before the
        # max_step_number was reached.
        recent[:] = (value for value in recent if value != 0)

        recent = recent[:-1]  # cannot visit current point again

        if num == 0:
            recent = []
        elif len(recent) > num:
            recent = recent[-num:]

        for i, pt in enumerate(points, start=0):
            if tuple(pt) in recent:
                points[i] = 0
        points[:] = (value for value in points if value != 0)
        return points

    def select_next_pose(self, points):
        """
        Select a random next pose and return as a tuple.

        :param points: a final 2D list of [x,y] points that are feasible next moves

        :return: a random (x,y) point from points
        """
        index = randint(0, len(points) - 1)
        next_pose = tuple(points[index])
        return next_pose

    def calculate_path(self, current_pose, path):
        """
        Determine if a path was found and return the result.

        :param current_pose: the current pose (x,y) of the robot
        :param path: a list of (x,y) tuples that describe the random path up to this point

        :return: None or a list of (x,y) tuples that describe a feasible path from the robot_pose to the goal_pose
        """
        if current_pose == self.gp:
            path[:] = (value for value in path if value != 0)
        else:
            path = None
        return path


class OptimalPlanner(DiscretePlanner):
    """ An optimal planner using A* search."""

    def search(self, world_state, robot_pose, goal_pose):
        """ 
        Execute the optimal planner.

        :type world_state: 2D list
        :param world_state: represents the world state, where 0 indicates a navigable space and 1 indicates an occupied/obstacle space

        :type robot_pose: tuple
        :param robot_pose: represents the robot pose (x,y) in the world state

        :type goal_pose: tuple
        :param goal_pose: represents the goal pose (x,y) in the world state

        :return: None or a list of (x,y) tuples that describe a shortest path from the robot_pose to the goal_pose
        """
        self.set_attributes(world_state, robot_pose, goal_pose)

        x = self.x_lim + 1
        y = self.y_lim + 1
        num_points = x * y

        # We begin by initializing three matrices that hold different
        # information about each point in the world space. The first matrix
        # holds information about the cost to reach each point. The next matrix
        # holds information about whether each point has been visited. The
        # third matrix holds information about the least costly path to each
        # point.
        cost_matrix = np.full((x, y), np.inf)
        visited = np.ones((x, y))
        unvisited = num_points
        path_matrix = np.zeros((x, y))
        path_matrix = np.ndarray.tolist(path_matrix)
        cost_matrix[self.rp[0]][self.rp[1]] = 0.00001       # close to zero

        while (unvisited > 0):
            # Find the next point and check if we have reached the goal or if
            # there is no possible path. If neither have happened, set the
            # current point to be visited and proceed to investigate neighbors.
            [cp, cp_cost] = self.select_current_point(cost_matrix, visited)

            if isinf(cp_cost):
                path = None
                break
            if tuple(cp) == self.gp:
                path = self.calculate_path(path_matrix, num_points)
                break

            visited[cp[0]][cp[1]] = np.inf

            # Generate a list of feasible orthogonal neighbors to the current
            # point. Check to see if reaching these neighbors through the
            # current point is the current least costly option
            list0 = self.get_orthogonal_points(cp)
            list1 = self.check_boundary(list0)
            list2 = self.check_obstacles(list1)

            self.calculate_cost(
                list2, cp, cp_cost,
                visited, cost_matrix, path_matrix,
            )

            unvisited -= 1

        return path

    def select_current_point(self, cost_matrix, visited):
        """
        Select an unvisited point with the lowest cost.

        :type cost_matrix: numpy.ndarray
        :param cost_matrix: contains the current minimum cost of reaching each point

        :type visited: numpy.ndarray
        :param visited: contains the visited status of each point

        :return: the selected current point (x,y) as a tuple and its current minimum cost
         """

        mat = np.multiply(cost_matrix, visited)
        cp_cost = mat.min()
        cp = tuple(np.unravel_index(mat.argmin(), mat.shape))

        return cp, cp_cost

    def calculate_cost(self, neighbors, current_point,
                       current_point_cost, visited, cost_matrix, path_matrix):
        """
        Calculate the cost of reaching each unvisited neighbors from the current point. Determine whether these paths are less costly than the current cost to reach each unvisited neighbor.
        
        :type neighbors: 2D list
        :param neighbors: feasible and orthogonal [x,y] points from the current_point

        :type current_point: tuple
        :param current_point: the current point (x,y)

        :type current_point_cost: float
        :param current_point_cost: minimum cost of reaching the current point

        :type visited: numpy.ndarray
        :param visited: contains the visited status of each point

        :type cost_matrix: numpy.ndarray
        :param cost_matrix: contains the current minimum cost of reaching each point

        :type path_matrix: numpy.ndarray
        :param path_matrix: contains the current shortest path to each point

        :return: updated cost_matrix and path_matrix
        """
        for n in neighbors:
            if visited[n[0]][n[1]] == 1.0:  # only unvisited neighbors

                # The heuristic function tells A* an estimate of the minimum
                # cost from any vertex n to the goal. I choose to use the
                # Manhattan Distance for square grid that allows movement in
                # only 4 directions.
                heuristic = abs(n[0] - self.gp[0]) + abs(n[1] - self.gp[1])

                # The cost of moving orthogonally in our square grid is always
                # 1
                unit_cost = 1

                # Calculate the total cost of moving to this neighbor from the
                # current point.
                alternate_cost = current_point_cost + heuristic + unit_cost

                # Determine if this alternate cost is less costly than current
                # cost for neighbor in the cost_matrix. If it is, store this
                # alternate cost in the cost_matrix and the current point in
                # the path_matrix.
                if alternate_cost < cost_matrix[n[0]][n[1]]:
                    cost_matrix[n[0]][n[1]] = alternate_cost
                    path_matrix[n[0]][n[1]] = current_point

        return cost_matrix, path_matrix

    def calculate_path(self, path_matrix, number):
        """
        Convert the path_matrix to a list of (x,y) tuples describing a path from the robot_pose and the goal_pose.

        :type path_matrix: numpy.ndarray
        :param path_matrix: contains the shortest path to each point

        :type number: int>=1
        :param number: total number of points in the world space

        :return: a list of (x,y) tuples that describe the shortest path between the robot_pose and the goal_pose
        """
        p = self.gp         # start at the goal_state and work backwards
        path = [0 for value in range(number)]
        path[0] = p
        step = 1

        while (p != self.rp):
            next_p = path_matrix[p[0]][p[1]]
            path[step] = next_p
            p = next_p
            step += 1

        path[:] = (value for value in path if value != 0)
        path = list(reversed(path))
        return path

