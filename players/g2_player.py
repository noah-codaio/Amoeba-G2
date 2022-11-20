import logging
import os
import pickle
from collections import deque
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import constants
from amoeba_state import AmoebaState

turn = 0


def to_cartesian(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def neighbors(coord, edge=constants.map_dim):
    x, y = coord
    return [
        ((x + dx) % edge, (y + dy) % edge)
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]
    ]


def find_movable_neighbor(map_state, x, y):
    out = []
    if map_state[x][(y - 1) % constants.map_dim] < 1:
        out.append((x, (y - 1) % constants.map_dim))
    if map_state[x][(y + 1) % constants.map_dim] < 1:
        out.append((x, (y + 1) % constants.map_dim))
    if map_state[(x - 1) % constants.map_dim][y] < 1:
        out.append(((x - 1) % constants.map_dim, y))
    if map_state[(x + 1) % constants.map_dim][y] < 1:
        out.append(((x + 1) % constants.map_dim, y))

    return out


def check_move(map_state, retract, move, periphery):
    if not set(retract).issubset(set(periphery)):
        return False

    movable = retract[:]
    new_periphery = list(set(periphery).difference(set(retract)))
    for i, j in new_periphery:
        nbr = find_movable_neighbor(map_state, i, j)
        for x, y in nbr:
            if (x, y) not in movable:
                movable.append((x, y))

    if not set(move).issubset(set(movable)):
        return False

    amoeba = np.copy(map_state)
    amoeba[amoeba < 0] = 0
    amoeba[amoeba > 0] = 1

    for i, j in retract:
        amoeba[i][j] = 0

    for i, j in move:
        amoeba[i][j] = 1

    tmp = np.where(amoeba == 1)
    result = list(zip(tmp[0], tmp[1]))
    check = np.zeros((constants.map_dim, constants.map_dim), dtype=int)

    def update(c):
        nonlocal check
        check[c] = 1

    traverse(amoeba, update)
    return (amoeba == check).all()


def torus_distance(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return np.sqrt(
        min(np.abs(x2 - x1), constants.map_dim - np.abs(x2 - x1)) ** 2
        + min(np.abs(y2 - y1), constants.map_dim - np.abs(y2 - y1)) ** 2
    )


def updated_amoeba_map(amoeba_map, retractions, additions):
    updated_amoeba = np.copy(amoeba_map)

    for i, j in retractions:
        updated_amoeba[i][j] = 0

    for i, j in additions:
        updated_amoeba[i][j] = 1

    return updated_amoeba


def map_to_coords(amoeba_map) -> list[tuple[int, int]]:
    return list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))


def center_of_mass(amoeba_map: npt.NDArray) -> tuple[float, float]:
    tiled_map = np.tile(amoeba_map, (2, 2))

    all_coords = map_to_coords(tiled_map)
    tiled_center = np.array([constants.map_dim, constants.map_dim])
    traversal_start = sorted(
        all_coords, key=lambda c: np.linalg.norm(c - tiled_center)
    )[0]

    seen_points = set()

    def update_map(c):
        nonlocal seen_points
        seen_points.add(c)

    traverse(tiled_map, update_map, traversal_start)
    filtered_coords = [coord for coord in all_coords if coord in seen_points]
    xs, ys = zip(*filtered_coords)

    cx = np.average(xs)
    cy = np.average(ys)

    return cx % constants.map_dim, cy % constants.map_dim


def traverse(
    amoeba_map: npt.NDArray,
    callback: Callable[[tuple[int, int]], None],
    start_point: Optional[tuple[int, int]] = None,
):
    amoeba_coords = set(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))
    if start_point is None:
        start_point = next(iter(amoeba_coords))
    stack = [start_point]
    seen = set()
    while len(stack):
        coord = stack.pop()
        if coord in seen:
            continue

        seen.add(coord)
        callback(coord)

        for neighbor in neighbors(coord, len(amoeba_map)):
            if neighbor not in seen and neighbor in amoeba_coords:
                stack.append(neighbor)


class Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        metabolism: float,
        goal_size: int,
        precomp_dir: str,
    ) -> None:
        """Initialise the player with the basic amoeba information

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
            logger (logging.Logger): logger use this like logger.info("message")
            metabolism (float): the percentage of amoeba cells, that can move
            goal_size (int): the size the amoeba must reach
            precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.metabolism = metabolism
        self.goal_size = goal_size

    def move(self, last_percept, current_percept, info) -> (list, list, int):
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement

        Args:
            last_percept (AmoebaState): contains state information after the previous move
            current_percept(AmoebaState): contains current state information
            info (int): byte (ranging from 0 to 256) to convey information from previous turn
        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]: This function returns three variables:
                1. A list of cells on the periphery that the amoeba retracts
                2. A list of positions the retracted cells have moved to
                3. A byte of information (values range from 0 to 255) that the amoeba can use
        """
        # Debugging
        global turn
        turn += 1

        amoeba_xs, amoeba_ys = current_percept.amoeba_map.nonzero()
        amoeba_coords = [(x, y) for x, y in np.transpose([amoeba_xs, amoeba_ys])]

        # print("Amoeba:", amoeba_coords)
        center = center_of_mass(current_percept.amoeba_map)

        delta = to_cartesian(min(current_percept.current_size / 2, 50), 3 * np.pi / 4)
        target_point = (center + delta) % 100
        print("Size", current_percept.current_size)
        print("Center", center)
        print("Delta", delta)
        print("Target", target_point)

        plt.clf()
        plt.scatter(amoeba_xs, amoeba_ys)
        plt.gca().set_xlim([0, 100])
        plt.gca().set_ylim([0, 100])
        plt.gca().invert_yaxis()
        plt.plot(center[0], center[1], color="green", marker="*")
        plt.plot(target_point[0], target_point[1], color="red", marker="v")
        plt.savefig(f"debug/{turn}.png", dpi=300)

        valid_additions_per_cell = {
            border_cell: [
                neighbor
                for neighbor in neighbors(border_cell)
                if neighbor not in current_percept.bacteria
                and neighbor not in amoeba_coords
            ]
            for border_cell in current_percept.periphery
        }

        # Addition target -> existing cells bordering it
        additions_with_support = {}
        for border_cell, cell_neighbors in valid_additions_per_cell.items():
            for neighbor in cell_neighbors:
                if not neighbor in additions_with_support:
                    additions_with_support[neighbor] = []
                additions_with_support[neighbor].append(border_cell)

        prioritized_targets = list(
            sorted(
                additions_with_support.keys(),
                key=lambda c: torus_distance(c, target_point),
            )
        )

        retractions_with_support = {
            border_cell: [
                neighbor
                for neighbor in neighbors(border_cell)
                if neighbor in amoeba_coords
            ]
            for border_cell in current_percept.periphery
        }
        prioritized_retractions = list(
            reversed(
                sorted(
                    current_percept.periphery,
                    key=lambda c: torus_distance(c, target_point),
                )
            )
        )

        retractions = []
        additions = []

        target_index = 0
        retraction_index = 0
        available_moves = int(np.ceil(self.metabolism * current_percept.current_size))
        for _ in range(available_moves):
            if retraction_index >= len(prioritized_retractions) or target_index >= len(
                prioritized_targets
            ):
                break
            retract = prioritized_retractions[retraction_index]
            addition = prioritized_targets[target_index]

            # Check if this addition is dependent on  a cell that was retracted
            if len(additions_with_support[addition]) == 0:
                target_index += 1
                continue

            # Check if this would split the amoeba
            retract_ok = True
            for neighbor in neighbors(retract):
                if neighbor in retractions_with_support:
                    if (
                        retract in retractions_with_support[neighbor]
                        and retractions_with_support[neighbor] == 1
                    ):
                        retract_ok = False

            # Check if this would cut off an addition we already locked in
            for existing_addition in additions:
                if (
                    retract in additions_with_support[existing_addition]
                    and len(additions_with_support[existing_addition]) == 1
                ):
                    retract_ok = False

            # If this retraction would remove the last support for this addition,
            # the addition wins
            if retract in additions_with_support[addition] and len(
                additions_with_support[addition]
            ):
                retract_ok = False

            if not retract_ok or not check_move(
                current_percept.amoeba_map,
                retractions + [retract],
                additions + [addition],
                current_percept.periphery,
            ):
                retraction_index += 1
                continue

            # Remove "support" from neighboring cells
            for neighbor in neighbors(retract):
                if neighbor in retractions_with_support:
                    retractions_with_support[neighbor] = [
                        support
                        for support in retractions_with_support[neighbor]
                        if support != retract
                    ]
                if neighbor in additions_with_support:
                    additions_with_support[neighbor] = [
                        support
                        for support in additions_with_support[neighbor]
                        if support != retract
                    ]

            retractions.append(retract)
            additions.append(addition)
            retraction_index += 1
            target_index += 1

        # print("Retract:", retractions)
        # print("Add:", additions)

        return retractions, additions, 0
