import unittest
from discrete_planner import DiscretePlanner, RandomPlanner, OptimalPlanner
from method_called import MethodCalled
import numpy as np


class TestDiscretePlanner(unittest.TestCase):
    """Tests for the class DiscretePlanner."""

    def setUp(self):
        """Create an instance of the class to use for all test methods."""
        self.my_planner = DiscretePlanner()

    def test_set_attributes_1(self):
        """Test the set_attributes method.

        Test that the world state, robot pose, and goal pose are being stored correctly as attributes.
        """
        ws = [
            [0, 0],
            [0, 1],
        ]
        rp = (0, 0)
        gp = (1, 1)
        self.my_planner.set_attributes(ws, rp, gp)

        self.assertEqual(self.my_planner.ws, ws)
        self.assertEqual(self.my_planner.rp, rp)
        self.assertEqual(self.my_planner.gp, gp)

    def test_set_attributes_2(self):
        """Test the set_attributes method.

        Test that x_lim and y_lim are stored correctly as attributes for a world state = [[0]].
        """
        ws = [[0]]
        rp = (0, 0)
        gp = (1, 1)
        x_lim = 1
        y_lim = 1

        self.my_planner.set_attributes(ws, rp, gp)

        self.assertEqual(self.my_planner.x_lim, x_lim)
        self.assertEqual(self.my_planner.y_lim, y_lim)

    def test_set_attributes_3(self):
        """Test the set_attributes method.

        Test that x_lim and y_lim are stored correctly as attributes for a world state = [[0, 1]].
        """
        ws = [[0, 1]]
        rp = (0, 0)
        gp = (1, 1)
        x_lim = 1
        y_lim = 2

        self.my_planner.set_attributes(ws, rp, gp)

        self.assertEqual(self.my_planner.x_lim, x_lim)
        self.assertEqual(self.my_planner.y_lim, y_lim)

    def test_set_attributes_4(self):
        """Test the set_attributes method.

        Test that x_lim and y_lim are stored correctly as attributes for a world state = [[0], [1]].
        """
        ws = [[0], [1]]
        rp = (0, 0)
        gp = (1, 1)
        x_lim = 2
        y_lim = 1

        self.my_planner.set_attributes(ws, rp, gp)

        self.assertEqual(self.my_planner.x_lim, x_lim)
        self.assertEqual(self.my_planner.y_lim, y_lim)

    def test_set_attributes_5(self):
        """Test the set_attributes method.

        Test that ws_b is calculated correctly and stored as an attribute.
        """
        ws = [
            [0, 1],
            [1, 0],
        ]
        rp = (0, 0)
        gp = (1, 1)
        ws_b = [[False, True], [True, False]]

        self.my_planner.set_attributes(ws, rp, gp)

        a = (self.my_planner.ws_b == ws_b)
        self.assertTrue(a.all())

    def test_get_orthogonal_points(self):
        """Test the get_orthogonal_points method."""
        cp = (0, 0)
        points = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        self.assertItemsEqual(
            self.my_planner.get_orthogonal_points(cp), points,
        )

    def test_check_boundary_1(self):
        """Test the check_boundary method.

        Test that points with negative coordinates are not passed.
        """
        self.my_planner.x_lim = 1
        self.my_planner.y_lim = 1
        points = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        points_bound = [[1, 0], [0, 1]]

        self.assertItemsEqual(
            self.my_planner.check_boundary(points), points_bound,
        )

    def test_check_boundary_2(self):
        """Test check_boundary method.

        Test that points with coordinates that exceed x_lim or y_lim are not passed.
        """
        self.my_planner.x_lim = 1
        self.my_planner.y_lim = 1
        points = [[2, 2], [1, 2], [0, 1], [1, 0], [2, 1]]
        points_bound = [[1, 0], [0, 1]]

        self.assertItemsEqual(
            self.my_planner.check_boundary(points), points_bound,
        )

    def test_check_obstacles_1(self):
        """Test the check_obstacle method.

        Test that check_obstacles returns correct values for the following world state: [[0]].
        """
        ws = [[0]]
        rp = (0, 0)
        gp = (1, 1)
        all_points = [[0, 0], [1, 0], [0, 1], [1, 1]]
        obs_free = all_points

        self.my_planner.set_attributes(ws, rp, gp)

        self.assertItemsEqual(
            self.my_planner.check_obstacles(all_points), obs_free,
        )

    def test_check_obstacles_2(self):
        """Test the check_obstacle method.

        Test that check_obstacles returns correct values for the following world state: [[1]].
        """
        ws = [[1]]
        rp = (0, 0)
        gp = (1, 1)
        all_points = [[0, 0], [1, 0], [0, 1], [1, 1]]
        obs_free = []

        self.my_planner.set_attributes(ws, rp, gp)

        self.assertItemsEqual(
            self.my_planner.check_obstacles(all_points), obs_free,
        )

    def test_check_obstacles_3(self):
        """Test the check_obstacle method.

        Test that check_obstacles returns correct values for the following world state:[[0],[1]].
        """
        ws = [[0], [1]]
        rp = (0, 0)
        gp = (1, 1)
        all_points = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
        obs_free = [[0, 0], [0, 1]]

        self.my_planner.set_attributes(ws, rp, gp)

        self.assertItemsEqual(
            self.my_planner.check_obstacles(all_points), obs_free,
        )

    def test_check_obstacles_4(self):
        """Test the check obstacle method.

        Test that check_obstacles returns correct values for the following world state: [[1, 0]].
        """
        ws = [[1, 0]]
        rp = (0, 0)
        gp = (1, 1)
        all_points = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
        obs_free = [[0, 2], [1, 2]]

        self.my_planner.set_attributes(ws, rp, gp)

        self.assertItemsEqual(
            self.my_planner.check_obstacles(all_points), obs_free,
        )

    def test_check_obstacles_5(self):
        """Test the check_obstacle method.

        Test that check_obstacles returns correct values for the following world state: [[0, 0], [1, 0]].
        """
        ws = [
            [0, 0],
            [1, 0],
        ]
        rp = (0, 0)
        gp = (1, 1)
        all_points = [
            [0, 0], [0, 1], [0, 2], [1, 0],
            [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]
        ]
        obs_free = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]

        self.my_planner.set_attributes(ws, rp, gp)
        self.assertItemsEqual(
            self.my_planner.check_obstacles(all_points), obs_free,
        )


class TestRandomPlanner(unittest.TestCase):
    """ Tests for the class RandomPlanner."""

    def setUp(self):
        """Create an instance of the class to use for all test methods."""
        self.my_planner = RandomPlanner()

    def test_set_max_step_number(self):
        """Test the set_max_step_number method."""
        msn = 50
        self.my_planner.set_max_step_number(msn)
        self.assertEqual(self.my_planner.msn, msn)

    def test_search_1(self):
        """Test the search method.

        Test when the robot is initially surrounded by the boundary and obstacle space. It should enter the if statement with the break command, never call the select_next_pose method, and the path variable should return None.
        """
        world_state = [
            [0, 1],
            [1, 0],
        ]
        robot_pose = (0, 0)
        goal_pose = (2, 2)
        msn = 2

        self.my_planner.select_next_pose = MethodCalled(
            self.my_planner.select_next_pose,
        )

        self.my_planner.set_max_step_number(msn)
        path = self.my_planner.search(world_state, robot_pose, goal_pose)

        self.assertFalse(self.my_planner.select_next_pose.was_called)
        self.assertEqual(path, None)

    def test_search_2(self):
        """Test the search method.

        Test when the max_step_number is less than what is needed to find the fastest path. The search method should always return None.
        """

        world_state = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
        robot_pose = (0, 0)
        goal_pose = (3, 3)

        # The fastest path has 7 points. For the random planner to find a path,
        # it would need to step at least 6 times.
        self.my_planner.set_max_step_number(5)

        for i in range(100):
            path = self.my_planner.search(world_state, robot_pose, goal_pose)
            if path:
                self.assertTrue(False)

    def test_search_3(self):
        """Test search method.

        Test when the max_step_number is equal to what is needed to find than the fastest path. There should be some run that will return the fastest path.
        """
        world_state = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
        robot_pose = (0, 0)
        goal_pose = (3, 3)

        # The fastest path has 7 points. For the random planner to find a path,
        # it would need to step at least 6 times.
        self.my_planner.set_max_step_number(6)

        found_path_at_least_once = False
        path_length = None
        for i in range(100):
            path = self.my_planner.search(world_state, robot_pose, goal_pose)
            if path:
                found_path_at_least_once = True
                path_length = len(path)

        self.assertTrue(found_path_at_least_once)
        self.assertEqual(path_length, 7)

    def test_check_recent_1(self):
        """Test the check_recent method.

        Test it gives the correct result when the max_step_number = 0.
        """
        self.my_planner.msn = 0
        recent = [
            (3, 2), (3, 3), (2, 3), (1, 3),
            (1, 2), (1, 1), (2, 1), (2, 2),
        ]
        pts = [(1, 2), (3, 2), (2, 1), (2, 3)]
        new_pts = pts

        self.assertItemsEqual(
            self.my_planner.check_recent(pts[:], recent), new_pts,
        )

    def test_check_recent_2(self):
        """Test the check_recent method.

        Test it gives the correct result when the max_step_numbers between 1 and 9 - sqrt(msn) is between 1 and 3.
        """
        recent = [
            (3, 2), (3, 3), (2, 3), (1, 3),
            (1, 2), (1, 1), (2, 1), (2, 2),
        ]
        pts = [(1, 2), (3, 2), (2, 1), (2, 3)]
        new_pts = [(1, 2), (3, 2), (2, 3)]

        for msn in range(1, 9):
            self.my_planner.msn = msn
            self.assertItemsEqual(
                self.my_planner.check_recent(pts[:], recent), new_pts,
            )

    def test_check_recent_3(self):
        """Test the check_recent method.

        Test it gives the correct result when the max_step_number is between 10 and 25 -  - sqrt(msn) is between 3 and 5.
        """
        recent = [
            (3, 2), (3, 3), (2, 3), (1, 3),
            (1, 2), (1, 1), (2, 1), (2, 2),
        ]
        pts = [(1, 2), (3, 2), (2, 1), (2, 3)]
        new_pts = [(3, 2), (2, 3)]

        for msn in range(10, 25):
            self.my_planner.msn = msn
            self.assertItemsEqual(
                self.my_planner.check_recent(pts[:], recent), new_pts,
            )

    def test_check_recent_4(self):
        """Test the check_recent method.

        Test it gives the correct result when the max_step_number is between 25 and 49 - sqrt(msn) is between 5 and 7.
        """
        recent = [
            (3, 2), (3, 3), (2, 3), (1, 3),
            (1, 2), (1, 1), (2, 1), (2, 2),
        ]
        pts = [(1, 2), (3, 2), (2, 1), (2, 3)]
        new_pts = [(3, 2)]

        for msn in range(25, 49):
            self.my_planner.msn = msn
            self.assertItemsEqual(
                self.my_planner.check_recent(pts[:], recent), new_pts,
            )

    def test_check_recent_5(self):
        """Test the check_recent method.

        Test it gives the correct result when the max_step_number = 49.
        """
        self.my_planner.msn = 49
        recent = [
            (3, 2), (3, 3), (2, 3), (1, 3),
            (1, 2), (1, 1), (2, 1), (2, 2),
        ]
        pts = [(1, 2), (3, 2), (2, 1), (2, 3)]
        new_pts = []

        self.assertItemsEqual(
            self.my_planner.check_recent(pts[:], recent), new_pts,
        )

    def test_select_next_pose(self):
        """Test the select_next_pose method."""
        pts = [[0, 0], [1, 1], [2, 2]]
        pts_tuple = [(0, 0), (1, 1), (2, 2)]
        next_pose = [0, 0, 0]

        for i in range(30):
            nxt = self.my_planner.select_next_pose(pts)
            if nxt == pts_tuple[0][:]:
                next_pose[0] = nxt
            elif nxt == pts_tuple[1][:]:
                next_pose[1] = nxt
            elif nxt == pts_tuple[2][:]:
                next_pose[2] = nxt
            else:
                self.assertTrue(False)

        self.assertItemsEqual(pts_tuple, next_pose)

    def test_calculate_path_1(self):
        """Test the calculate_path method.

        Test that it gives the correct path when the current pose equals the goal pose.
        """
        self.my_planner.gp = (2, 3)
        cp = (2, 3)
        path = [(1, 1), (2, 2), (3, 3), 0, 0, 0]
        final_path = [(1, 1), (2, 2), (3, 3)]

        self.assertEqual(self.my_planner.calculate_path(cp, path), final_path)

    def test_calculate_path_2(self):
        """Test the calculate_path method.

        Test that it gives None when the current pose does not equal the goal pose.
        """
        self.my_planner.gp = (2, 3)
        cp = (2, 4)
        path = [(1, 1), (2, 2), (3, 3), 0, 0, 0]

        self.assertEqual(self.my_planner.calculate_path(cp, path), None)


class TestOptimalPlanner(unittest.TestCase):
    """Tests for the OptimalPlanner class."""

    def setUp(self):
        """Create an instance of the class to use for all test methods."""
        self.my_planner = OptimalPlanner()

    def test_search_1(self):
        """Test the search method.

        Test to see that a shortest path is found for an environment with a feasible path.
        """
        world_state = [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        robot_pose = (0, 0)
        goal_pose = (0, 9)
        shortest_path_length = 30

        path = self.my_planner.search(world_state, robot_pose, goal_pose)
        self.assertEqual(len(path), shortest_path_length)

    def test_search_2(self):
        """Test the search method.

        Test to see that a path is not found for an environment where there is no feasible path.
        """
        world_state = [
            [0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        robot_pose = (0, 0)
        goal_pose = (0, 9)

        path = self.my_planner.search(world_state, robot_pose, goal_pose)
        self.assertEqual(path, None)

    def test_select_current_point_1(self):
        """Test the select_current_point method.

        Test that it returns the correct point when there is only one unvisited point with the lowest cost.
        """
        c_matrix = [
            [np.inf, 2],
            [1, np.inf],
        ]
        visited = [
            [np.inf, 1],
            [np.inf, 1],
        ]

        cp_test = (0, 1)
        cp_cost_test = 2

        [cp, cp_cost] = self.my_planner.select_current_point(c_matrix, visited)
        self.assertEqual(cp, cp_test)
        self.assertEqual(cp_cost, cp_cost_test)

    def test_select_current_point_2(self):
        """Test the select_current_point method.

        Test that it returns one of the correct points when there are multiple unvisited points with the lowest cost.
        """
        c_matrix = [
            [2, 2],
            [1, np.inf],
        ]
        visited = [
            [1, 1],
            [np.inf, 1],
        ]

        cp_test = [(0, 1), (0, 0)]
        cp_cost_test = 2

        [cp, cp_cost] = self.my_planner.select_current_point(c_matrix, visited)
        self.assertIn(cp, cp_test)
        self.assertEqual(cp_cost, cp_cost_test)

    def test_calculate_cost_1(self):
        """Test the calculate_cost method. 

        Test that only unvisited neighbors are looked at.
        """
        self.my_planner.gp = (1, 1)
        pts = [[0, 1], [1, 0]]
        cp = (0, 0)
        cp_cost = 0

        visited = [
            [1, np.inf],
            [1, 1],
        ]
        cost_matrix = [
            [cp_cost, np.inf],
            [np.inf, np.inf],
        ]
        path_matrix = [
            [0, 0],
            [0, 0],
        ]

        cost_matrix_test = [
            [cp_cost, np.inf],
            [2, np.inf],
        ]
        path_matrix_test = [
            [0, 0],
            [(0, 0), 0],
        ]

        [c_m, p_m] = self.my_planner.calculate_cost(
            pts, cp, cp_cost, visited,
            cost_matrix, path_matrix,
        )
        self.assertEqual(c_m, cost_matrix_test)
        self.assertEqual(p_m, path_matrix_test)

    def test_calculate_cost_2(self):
        """Test the calculate_cost method. 

        Test that the alternate cost is calculated and stored correctly. Test that the path_matrix is updated correctly.
        """
        self.my_planner.gp = (0, 0)
        pts = [[1, 0], [2, 1]]
        cp = (2, 0)
        cp_cost = 0.5

        visited = [
            [1, 1],
            [1, 1],
            [1, 1],
        ]
        cost_matrix = [
            [np.inf, np.inf],
            [np.inf, np.inf],
            [cp_cost, np.inf]
        ]
        path_matrix = [
            [0, 0],
            [0, 0],
            [0, 0],
        ]

        cost_matrix_test = [
            [np.inf, np.inf],
            [2.5, np.inf],
            [cp_cost, 4.5],
        ]
        path_matrix_test = [
            [0, 0],
            [(2, 0), 0],
            [0, (2, 0)],
        ]

        [c_m, p_m] = self.my_planner.calculate_cost(
            pts, cp, cp_cost, visited,
            cost_matrix, path_matrix,
        )
        self.assertEqual(c_m, cost_matrix_test)
        self.assertEqual(p_m, path_matrix_test)

    def test_calculate_cost_3(self):
        """Test the calculate_cost method. 

        Test that the alternate cost is only stored when it is less than the corresponding cost in the cost matrix. Test that the path_matrix is updated correctly.
        """
        self.my_planner.gp = (0, 0)
        pts = [[1, 0], [2, 1]]
        cp = (2, 0)
        cp_cost = 0.5

        visited = [
            [1, 1],
            [1, 1],
            [1, 1],
        ]
        cost_matrix = [
            [np.inf, np.inf],
            [3, np.inf],
            [cp_cost, 3],
        ]
        path_matrix = [
            [0, 0],
            [0, 0],
            [0, 0],
        ]

        cost_matrix_test = [
            [np.inf, np.inf],
            [2.5, np.inf],
            [cp_cost, 3],
        ]
        path_matrix_test = [
            [0, 0],
            [(2, 0), 0],
            [0, 0],
        ]

        [c_m, p_m] = self.my_planner.calculate_cost(
            pts, cp, cp_cost, visited,
            cost_matrix, path_matrix,
        )
        self.assertEqual(c_m, cost_matrix_test)
        self.assertEqual(p_m, path_matrix_test)

    def test_calculate_path(self):
        """Test the calculate_path method."""
        self.my_planner.rp = (2, 0)
        self.my_planner.gp = (0, 2)
        num = 3 * 3

        path_matrix = [
            [(1, 0), (0, 0), (0, 1)],
            [(2, 0), 0, 0],
            [0, 0, 0],
        ]
        test_path = [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]

        path = self.my_planner.calculate_path(path_matrix, num)
        self.assertEqual(test_path, path)

if __name__ == '__main__':
    unittest.main(exit=False)
