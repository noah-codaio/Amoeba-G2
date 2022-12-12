import logging
import math
import os
import pickle
import time
from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

import constants
from amoeba_state import AmoebaState

# ---------------------------------------------------------------------------- #
#                               Constants                                      #
# ---------------------------------------------------------------------------- #

ABLATION_STAGE = 4
ENABLE_VERTICAL_FLIP = False
ENABLE_FORMATION_THRESHOLD = False
ENABLE_EXTRA_MOVES = False
ENABLE_SORT_RETRACTS = False
if ABLATION_STAGE >= 1:
    ENABLE_VERTICAL_FLIP = True
if ABLATION_STAGE >= 2:
    ENABLE_FORMATION_THRESHOLD = True
if ABLATION_STAGE >= 3:
    ENABLE_EXTRA_MOVES = True
if ABLATION_STAGE >= 4:
    ENABLE_SORT_RETRACTS = True

CENTER_X = constants.map_dim // 2
CENTER_Y = constants.map_dim // 2

DEFAULT_TEETH_GAP = 2
DEFAULT_TEETH_SHIFT_PERIOD = 6

DEFAULT_FORMATION_THRESHOLD = 0.7

DEFAULT_VERTICAL_FLIP_SIZE = 100

DEFAULT_ONE_WIDE_BACKBONE = False


# Maping from (size, density) to the number of parameter value
TEETH_GAP_MAP = {
    (3, 0.05): 1,
    (3, 0.1): 1,
    (3, 0.25): 1,
    (3, 0.4): 1,
    (3, 1.0): 1,
    (5, 0.05): 1,
    (5, 0.1): 1,
    (5, 0.25): 1,
    (5, 0.4): 1,
    (5, 1.0): 1,
    (8, 0.05): 1,
    (8, 0.1): 1,
    (8, 0.25): 1,
    (8, 0.4): 1,
    (8, 1.0): 1,
    (15, 0.05): 1,
    (15, 0.1): 1,
    (15, 0.25): 1,
    (15, 0.4): 1,
    (15, 1.0): 1,
    (25, 0.05): 2,
    (25, 0.1): 2,
    (25, 0.25): 2,
    (25, 0.4): 2,
    (25, 1.0): 2,
}
TEETH_SHIFT_PERIOD_MAP = {
    (3, 0.05): 6,
    (3, 0.1): 6,
    (3, 0.25): 6,
    (3, 0.4): 6,
    (3, 1.0): 6,
    (5, 0.05): 6,
    (5, 0.1): 6,
    (5, 0.25): 6,
    (5, 0.4): 6,
    (5, 1.0): 6,
    (8, 0.05): 6,
    (8, 0.1): 6,
    (8, 0.25): 6,
    (8, 0.4): 6,
    (8, 1.0): 6,
    (15, 0.05): 6,
    (15, 0.1): 6,
    (15, 0.25): 6,
    (15, 0.4): 6,
    (15, 1.0): 6,
    (25, 0.05): 6,
    (25, 0.1): 6,
    (25, 0.25): 6,
    (25, 0.4): 6,
    (25, 1.0): 6,
}
FORMATION_THRESHOLD_MAP = {
    (3, 0.05): 1.0,
    (3, 0.1): 1.0,
    (3, 0.25): 1.0,
    (3, 0.4): 1.0,
    (3, 1.0): 1.0,
    (5, 0.05): 1.0,
    (5, 0.1): 1.0,
    (5, 0.25): 1.0,
    (5, 0.4): 1.0,
    (5, 1.0): 1.0,
    (8, 0.05): 1.0,
    (8, 0.1): 1.0,
    (8, 0.25): 1.0,
    (8, 0.4): 1.0,
    (8, 1.0): 1.0,
    (15, 0.05): 1.0,
    (15, 0.1): 1.0,
    (15, 0.25): 1.0,
    (15, 0.4): 1.0,
    (15, 1.0): 1.0,
    (25, 0.05): 1.0,
    (25, 0.1): 1.0,
    (25, 0.25): 1.0,
    (25, 0.4): 1.0,
    (25, 1.0): 1.0,
}
if ENABLE_FORMATION_THRESHOLD:
    FORMATION_THRESHOLD_MAP = {
        (3, 0.05): 0.5,
        (3, 0.1): 0.5,
        (3, 0.25): 0.5,
        (3, 0.4): 0.5,
        (3, 1.0): 0.5,
        (5, 0.05): 0.5,
        (5, 0.1): 0.5,
        (5, 0.25): 0.5,
        (5, 0.4): 0.5,
        (5, 1.0): 0.5,
        (8, 0.05): 0.5,
        (8, 0.1): 0.5,
        (8, 0.25): 0.5,
        (8, 0.4): 0.5,
        (8, 1.0): 0.5,
        (15, 0.05): 0.7,
        (15, 0.1): 0.7,
        (15, 0.25): 0.7,
        (15, 0.4): 0.7,
        (15, 1.0): 0.7,
        (25, 0.05): 0.8,
        (25, 0.1): 0.8,
        (25, 0.25): 0.8,
        (25, 0.4): 0.8,
        (25, 1.0): 0.8,
    }
VERTICAL_FLIP_SIZE_MAP = {
    (3, 0.05): 200,
    (3, 0.1): 200,
    (3, 0.25): 200,
    (3, 0.4): 200,
    (3, 1.0): 200,
    (5, 0.05): 200,
    (5, 0.1): 200,
    (5, 0.25): 200,
    (5, 0.4): 200,
    (5, 1.0): 200,
    (8, 0.05): 100,
    (8, 0.1): 100,
    (8, 0.25): 100,
    (8, 0.4): 100,
    (8, 1.0): 100,
    (15, 0.05): 100,
    (15, 0.1): 100,
    (15, 0.25): 100,
    (15, 0.4): 100,
    (15, 1.0): 100,
    (25, 0.05): 100,
    (25, 0.1): 100,
    (25, 0.25): 100,
    (25, 0.4): 100,
    (25, 1.0): 100,
}
ONE_WIDE_BACKBONE_MAP = {
    (3, 0.05): True,
    (3, 0.1): True,
    (3, 0.25): True,
    (3, 0.4): True,
    (3, 1.0): True,
    (5, 0.05): False,
    (5, 0.1): False,
    (5, 0.25): False,
    (5, 0.4): False,
    (5, 1.0): False,
    (8, 0.05): False,
    (8, 0.1): False,
    (8, 0.25): False,
    (8, 0.4): False,
    (8, 1.0): False,
    (15, 0.05): False,
    (15, 0.1): False,
    (15, 0.25): False,
    (15, 0.4): False,
    (15, 1.0): False,
    (25, 0.05): False,
    (25, 0.1): False,
    (25, 0.25): False,
    (25, 0.4): False,
    (25, 1.0): False,
}


class PlayerParameters:
    formation_threshold: float
    teeth_gap: int
    teeth_shift_period: int
    one_wide_backbone: bool
    vertical_flip_size: int

    def __init__(self):
        self.formation_threshold = DEFAULT_FORMATION_THRESHOLD
        self.teeth_gap = DEFAULT_TEETH_GAP
        self.teeth_shift_period = DEFAULT_TEETH_SHIFT_PERIOD
        self.one_wide_backbone = DEFAULT_ONE_WIDE_BACKBONE
        self.vertical_flip_size = DEFAULT_VERTICAL_FLIP_SIZE


default_params = PlayerParameters()

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #


def map_to_coords(amoeba_map: npt.NDArray) -> list[Tuple[int, int]]:
    return list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))


def coords_to_map(coords: list[tuple[int, int]], size=constants.map_dim) -> npt.NDArray:
    amoeba_map = np.zeros((size, size), dtype=np.int8)
    for x, y in coords:
        amoeba_map[x, y] = 1
    return amoeba_map


def show_amoeba_map(amoeba_map: npt.NDArray, retracts=[], extends=[], title="") -> None:
    retracts_map = coords_to_map(retracts)
    extends_map = coords_to_map(extends)

    map = np.zeros((constants.map_dim, constants.map_dim), dtype=np.int8)
    for x in range(constants.map_dim):
        for y in range(constants.map_dim):
            # transpose map for visualization as we add cells
            if retracts_map[x, y] == 1:
                map[y, x] = -1
            elif extends_map[x, y] == 1:
                map[y, x] = 2
            elif amoeba_map[x, y] == 1:
                map[y, x] = 1

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.pcolormesh(map, edgecolors="k", linewidth=0.1)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.title(title)
    # plt.show()
    plt.savefig(f"formation_map/{time.time() * 1000}.png")


# ---------------------------------------------------------------------------- #
#                                Memory Bit Mask                               #
# ---------------------------------------------------------------------------- #


class MemoryFields(Enum):
    VerticalInvert = 0


def read_memory(memory: int) -> dict[MemoryFields, bool]:
    out = {}
    for field in MemoryFields:
        value = True if (memory & (1 << field.value)) >> field.value else False
        out[field] = value
    return out


def change_memory_field(memory: int, field: MemoryFields, value: bool) -> int:
    bit = 1 if value else 0
    mask = 1 << field.value
    # Unset the bit, then or in the new bit
    return (memory & ~mask) | ((bit << field.value) & mask)


# ---------------------------------------------------------------------------- #
#                               Formation Class                                #
# ---------------------------------------------------------------------------- #


class Formation:
    def __init__(self, initial_formation=None) -> None:
        self.map = (
            initial_formation
            if initial_formation
            else np.zeros((constants.map_dim, constants.map_dim), dtype=np.int8)
        )
        self.cells = np.count_nonzero(self.map)

    def add_cell(self, x, y) -> None:
        self.map[x % constants.map_dim, y % constants.map_dim] = 1
        self.cells += 1

    def get_cell(self, x, y) -> int:
        return self.map[x % constants.map_dim, y % constants.map_dim]

    def merge_formation(self, formation_map: npt.NDArray):
        self.map = np.logical_or(self.map, formation_map)
        self.cells = np.count_nonzero(self.map)


# ---------------------------------------------------------------------------- #
#                               Main Player Class                              #
# ---------------------------------------------------------------------------- #


class Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        metabolism: float,
        goal_size: int,
        precomp_dir: str,
        params: PlayerParameters = default_params,
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
        self.current_size = goal_size / 4
        self.params = params

        self.set_game_params()

        self.teeth_shift_list = (
            (
                [0 for i in range(self.params.teeth_shift_period)]
                + [1 for i in range(self.params.teeth_shift_period)]
            )
            * (round(np.ceil(100 / (self.params.teeth_shift_period * 2))))
        )[:100]

        # Class accessible percept variables, written at the start of each turn
        self.current_size: int = None
        self.amoeba_map: npt.NDArray = None
        self.bacteria_cells: set[Tuple[int, int]] = None
        self.retractable_cells: List[Tuple[int, int]] = None
        self.extendable_cells: List[Tuple[int, int]] = None
        self.num_available_moves: int = None

    def set_game_params(self) -> None:
        start_size = math.sqrt(self.current_size)
        self.params.teeth_gap = TEETH_GAP_MAP.get(
            (start_size, self.metabolism), DEFAULT_TEETH_GAP
        )
        self.params.teeth_shift_period = TEETH_SHIFT_PERIOD_MAP.get(
            (start_size, self.metabolism), DEFAULT_TEETH_SHIFT_PERIOD
        )
        self.params.formation_threshold = FORMATION_THRESHOLD_MAP.get(
            (start_size, self.metabolism), DEFAULT_FORMATION_THRESHOLD
        )
        self.params.vertical_flip_size = VERTICAL_FLIP_SIZE_MAP.get(
            (start_size, self.metabolism), DEFAULT_VERTICAL_FLIP_SIZE
        )
        self.params.one_wide_backbone = ONE_WIDE_BACKBONE_MAP.get(
            (start_size, self.metabolism), DEFAULT_ONE_WIDE_BACKBONE
        )

    def generate_comb_formation(
        self,
        size: int,
        tooth_offset=0,
        center_x=CENTER_X,
        center_y=CENTER_Y,
        comb_idx=0,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate a comb formation of a given size, returning a tuple of the formation map and the bridge map"""

        comb_formation = Formation()
        bridge_formation = Formation()
        comb_0_center_x = center_x

        if size < 2:
            return comb_formation.map, bridge_formation.map

        one_wide_backbone = True if size < 36 else self.params.one_wide_backbone
        if not one_wide_backbone:
            teeth_size = min(round(size / ((self.params.teeth_gap + 1) * 2 + 1)), 49)
            backbone_size = min((size - teeth_size) // 2, 99)
        else:
            teeth_size = min(round(size / ((self.params.teeth_gap + 1) + 1)), 49)
            backbone_size = min((size - teeth_size), 99)
        cells_used = backbone_size * 2 + teeth_size

        # If we have hit our max size, form an additional comb and connect it via a bridge
        if backbone_size == 99 and comb_idx == 0:
            comb_0_x_offset = center_x - CENTER_X
            comb_1_center_x = CENTER_X - comb_0_x_offset

            # Bridge between the two combs
            for i in range(100):
                if size - cells_used > 0:
                    bridge_formation.add_cell(
                        (comb_0_center_x - i) % constants.map_dim, center_y
                    )
                    cells_used += 1

            # Generate the second comb
            if size - cells_used > 0:
                second_comb, second_bridge = self.generate_comb_formation(
                    size - cells_used, tooth_offset, comb_1_center_x, center_y, 1
                )
                comb_formation.merge_formation(second_comb)
                bridge_formation.merge_formation(second_bridge)

        # Build first comb formation

        # Add center cells
        comb_formation.add_cell(comb_0_center_x, center_y)
        comb_formation.add_cell(comb_0_center_x - 1, center_y)

        # Then prioritize adding teeth
        for i in range(
            1,
            round(
                min((teeth_size * (self.params.teeth_gap + 1)) / 2, backbone_size / 2)
                + 0.1
            ),
            self.params.teeth_gap + 1,
        ):
            comb_formation.add_cell(
                comb_0_center_x + (1 if comb_idx == 0 else -1),
                center_y + tooth_offset + i,
            )
            comb_formation.add_cell(
                comb_0_center_x + (1 if comb_idx == 0 else -1),
                center_y + tooth_offset - i,
            )

        # Then build the backbone (we may not have quite enough cells for this)
        for i in range(1, round((backbone_size - 1) / 2 + 0.1) + 1):
            # first layer of backbone
            comb_formation.add_cell(comb_0_center_x, center_y + i)
            comb_formation.add_cell(comb_0_center_x, center_y - i)
            # second layer of backbone
            if not one_wide_backbone:
                if comb_formation.cells < size:
                    comb_formation.add_cell(
                        comb_0_center_x + (-1 if comb_idx == 0 else 1), center_y + i
                    )
                if comb_formation.cells < size:
                    comb_formation.add_cell(
                        comb_0_center_x + (-1 if comb_idx == 0 else 1), center_y - i
                    )

        # If we build a second comb, build up additional cells in the center
        if backbone_size == 99 and comb_idx == 0:
            cells_remaining = (
                size
                - np.count_nonzero(comb_formation.map)
                - np.count_nonzero(bridge_formation.map)
            )
            bridge_offset = 1
            while cells_remaining > 0 and bridge_offset < 99:
                for i in range(100):
                    y_offset = (
                        bridge_offset if bridge_offset <= 49 else 50 - bridge_offset
                    )
                    x_position = (comb_0_center_x - i) % constants.map_dim
                    if comb_formation.get_cell(x_position, center_y + y_offset) == 0:
                        bridge_formation.add_cell(x_position, center_y + y_offset)
                        cells_remaining -= 1
                        if cells_remaining <= 0:
                            break
                bridge_offset += 1

        # show_amoeba_map(comb_formation.map, title="Generated Comb Formation")
        # show_amoeba_map(bridge_formation.map, title="Generated Bridge Formation")
        # show_amoeba_map(comb_formation.map + bridge_formation.map, title="Generated Formation")
        return comb_formation.map, bridge_formation.map

    def get_morph_moves(
        self, desired_amoeba: npt.NDArray, center_y=CENTER_Y
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Function which takes a starting amoeba state and a desired amoeba state and generates a set of retracts and extends
        to morph the amoeba shape towards the desired shape.
        """

        current_points = map_to_coords(self.amoeba_map)
        desired_points = map_to_coords(desired_amoeba)

        # Sort retracts based on distance from formation. Reduces straggling branches lagging behind formation.
        potential_retracts = [
            p
            for p in list(set(current_points).difference(set(desired_points)))
            if p in self.retractable_cells
        ]
        if ENABLE_SORT_RETRACTS:
            kdtree = KDTree(desired_points)
            potential_retracts.sort(
                reverse=True, key=lambda p: kdtree.query([p], k=1)[0]
            )

        potential_extends = [
            p
            for p in list(set(desired_points).difference(set(current_points)))
            if p in self.extendable_cells
        ]
        # potential_extends.sort(key=lambda p: p[1])

        # show_amoeba_map(desired_amoeba, title="Desired Amoeba")
        # show_amoeba_map(self.amoeba_map, potential_retracts, potential_extends, title="Current Amoeba, Potential Retracts and Extends")

        # Loop through potential extends, searching for a matching retract
        retracts = []
        extends = []
        error_ext = []
        error_ret = []

        check_calls = 0
        SKIP_PER = 0.25
        skip_n = int(
            SKIP_PER * min(len(potential_extends), len(potential_retracts))
        )  # Scales with size of amoeba
        skip_n = min(skip_n, 200)  # Cap it at 200
        if skip_n < 1:
            skip_n = 1
        count = 0
        for potential_extend, potential_retract in zip(
            potential_extends, potential_retracts
        ):
            # Ensure we only move as much as possible given our current metabolism
            if len(extends) >= self.num_available_moves:
                break

            extends.append(potential_extend)
            retracts.append(potential_retract)

            # Check if the last N calls are okay
            count += 1
            if count % skip_n == 0:
                while not self.check_move(retracts, extends):
                    # remove elements one-by-one till it works
                    check_calls += 1
                    error_ext.append(extends.pop())
                    error_ret.append(retracts.pop())

        while not self.check_move(retracts, extends):
            # remove elements one-by-one till it works
            check_calls += 1
            error_ext.append(extends.pop())
            error_ret.append(retracts.pop())

        # Shorten the original lists for later use
        potential_extends = list(set(potential_extends) - set(extends))
        potential_retracts = list(set(potential_retracts) - set(retracts))

        # For the left-over error causing retract/extends, try again with a thorough search
        if potential_extends and potential_retracts:
            for pot_ext in potential_extends:
                # Ensure we only move as much as possible given our current metabolism
                if len(extends) >= self.num_available_moves:
                    break

                for pot_ret in potential_retracts:
                    if self.check_move(retracts + [pot_ret], extends + [pot_ext]):
                        check_calls += 1
                        retracts.append(pot_ret)
                        extends.append(pot_ext)

                        try:
                            potential_extends.remove(pot_ext)
                        except ValueError:
                            pass
                        try:
                            potential_retracts.remove(pot_ret)
                        except ValueError:
                            pass
                        break

        # If we have moves remaining, 'store' the remaining extends and retracts in the center of the amoeba
        if (
            len(retracts) < self.num_available_moves
            and len(potential_retracts) > 0
            and len(extends) > 0
            and self.current_size > 16
            and ENABLE_EXTRA_MOVES
        ):
            potential_extends = [
                p
                for p in self.extendable_cells
                if p not in retracts and p not in extends
            ]
            potential_extends.sort(key=lambda p: np.absolute(center_y - p[1]))
            potential_retracts.sort(
                key=lambda p: np.absolute(center_y - p[1]), reverse=True
            )

            # show_amoeba_map(self.amoeba_map, retracts, extends, "Planned")
            # show_amoeba_map(self.amoeba_map, potential_retracts, potential_extends, "Possible Remaining")

            for potential_extend in potential_extends:
                if len(extends) >= self.num_available_moves:
                    break

                for potential_retract in potential_retracts:
                    if np.absolute(center_y - potential_extend[1]) < np.absolute(
                        center_y - potential_retract[1]
                    ) and self.check_move(
                        retracts + [potential_retract], extends + [potential_extend]
                    ):
                        check_calls += 1
                        retracts.append(potential_retract)
                        extends.append(potential_extend)
                        potential_retracts.remove(potential_retract)
                        break

        # show_amoeba_map(self.amoeba_map, retracts, extends, title="Current Amoeba, Selected Retracts and Extends")
        print(f"Check calls: {check_calls} / {self.current_size}")

        # retracts = list(set(retracts))
        # extends = list(set(extends))
        return retracts, extends

    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria, mini):
        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        movable += retract

        return movable[:mini]

    def find_movable_neighbor(
        self, x: int, y: int, amoeba_map: npt.NDArray, bacteria: set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        out = []
        if (x, y) not in bacteria:
            if amoeba_map[x][(y - 1) % constants.map_dim] == 0:
                out.append((x, (y - 1) % constants.map_dim))
            if amoeba_map[x][(y + 1) % constants.map_dim] == 0:
                out.append((x, (y + 1) % constants.map_dim))
            if amoeba_map[(x - 1) % constants.map_dim][y] == 0:
                out.append(((x - 1) % constants.map_dim, y))
            if amoeba_map[(x + 1) % constants.map_dim][y] == 0:
                out.append(((x + 1) % constants.map_dim, y))
        return out

    # Adapted from amoeba_game code
    def check_move(
        self, retracts: List[Tuple[int, int]], extends: List[Tuple[int, int]]
    ) -> bool:
        if not set(retracts).issubset(set(self.retractable_cells)):
            return False

        movable = set(retracts[:])
        new_periphery = list(set(self.retractable_cells).difference(set(retracts)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, self.amoeba_map, self.bacteria_cells)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.add((x, y))

        if not set(extends).issubset(movable):
            return False

        amoeba = np.copy(self.amoeba_map)
        amoeba[amoeba < 0] = 0
        amoeba[amoeba > 0] = 1

        for i, j in retracts:
            amoeba[i][j] = 0

        for i, j in extends:
            amoeba[i][j] = 1

        tmp = np.where(amoeba == 1)
        result = list(zip(tmp[0], tmp[1]))
        check = np.zeros((constants.map_dim, constants.map_dim), dtype=int)

        stack = result[0:1]
        result = set(result)
        while len(stack):
            a, b = stack.pop()
            check[a][b] = 1

            if (a, (b - 1) % constants.map_dim) in result and check[a][
                (b - 1) % constants.map_dim
            ] == 0:
                stack.append((a, (b - 1) % constants.map_dim))
            if (a, (b + 1) % constants.map_dim) in result and check[a][
                (b + 1) % constants.map_dim
            ] == 0:
                stack.append((a, (b + 1) % constants.map_dim))
            if ((a - 1) % constants.map_dim, b) in result and check[
                (a - 1) % constants.map_dim
            ][b] == 0:
                stack.append(((a - 1) % constants.map_dim, b))
            if ((a + 1) % constants.map_dim, b) in result and check[
                (a + 1) % constants.map_dim
            ][b] == 0:
                stack.append(((a + 1) % constants.map_dim, b))

        return (amoeba == check).all()

    def store_current_percept(self, current_percept: AmoebaState) -> None:
        self.current_size = current_percept.current_size
        self.amoeba_map = current_percept.amoeba_map
        self.retractable_cells = current_percept.periphery
        self.bacteria_cells = set(current_percept.bacteria)
        self.extendable_cells = current_percept.movable_cells
        self.num_available_moves = int(
            np.ceil(self.metabolism * current_percept.current_size)
        )

        self.amoeba_map = np.bitwise_or(
            self.amoeba_map, coords_to_map(self.bacteria_cells)
        )

    def check_and_initialize_memory(self, memory: int) -> int:
        if (
            memory == 0
            and self.current_size == self.goal_size / 4
            and self.amoeba_map[50][50]
        ):
            return (CENTER_X + 1 if self.current_size < 36 else CENTER_X + 3) << 1
        return memory

    def move(
        self, last_percept: AmoebaState, current_percept: AmoebaState, info: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]:
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
        self.store_current_percept(current_percept)

        retracts = []
        moves = []

        info = self.check_and_initialize_memory(info)

        # Extract backbone column from memory
        curr_backbone_col = info >> 1

        # Alternate vertical translation direction if necessary
        memory_fields = read_memory(info)

        teeth_shift = self.teeth_shift_list[curr_backbone_col]
        curr_backbone_row = (
            curr_backbone_col
            if not memory_fields[MemoryFields.VerticalInvert]
            else constants.map_dim - curr_backbone_col
        )
        next_comb, next_bridge = self.generate_comb_formation(
            self.current_size,
            teeth_shift,
            curr_backbone_col,
            CENTER_Y
            # curr_backbone_row,
        )
        if (
            memory_fields[MemoryFields.VerticalInvert]
            and self.current_size < self.params.vertical_flip_size
        ):
            next_comb = np.rot90(next_comb)
            next_bridge = np.rot90(next_bridge)

        # Check if current comb formation is filled
        comb_mask = self.amoeba_map[next_comb.nonzero()]
        settled = (sum(comb_mask) / len(comb_mask)) > self.params.formation_threshold
        if not settled:
            retracts, moves = self.get_morph_moves(
                next_comb + next_bridge,
                CENTER_Y
                # curr_backbone_row
            )

            # Actually, we have no more moves to make
            if len(moves) == 0:
                settled = True

        if settled:
            # When we "settle" into the target backbone column, advance the backbone column by 1
            prev_backbone_col = curr_backbone_col
            prev_backbone_row = curr_backbone_row
            new_backbone_col = (prev_backbone_col + 1) % 100
            new_backbone_row = (
                new_backbone_col
                if not memory_fields[MemoryFields.VerticalInvert]
                else constants.map_dim - new_backbone_col
            )
            teeth_shift = self.teeth_shift_list[new_backbone_col]
            next_comb, next_bridge = self.generate_comb_formation(
                self.current_size,
                teeth_shift,
                prev_backbone_col,
                CENTER_Y
                # new_backbone_row,
            )

            if (
                memory_fields[MemoryFields.VerticalInvert]
                and self.params.vertical_flip_size
                and ENABLE_VERTICAL_FLIP
            ):
                next_comb = np.rot90(next_comb)
                next_bridge = np.rot90(next_bridge)

            retracts, moves = self.get_morph_moves(
                next_comb + next_bridge,
                CENTER_Y
                # curr_backbone_row
            )

            if curr_backbone_col == 50:
                info = change_memory_field(
                    info,
                    MemoryFields.VerticalInvert,
                    not memory_fields[MemoryFields.VerticalInvert]
                    if self.current_size < self.params.vertical_flip_size
                    else 0,
                )
                memory_fields = read_memory(info)

            info = new_backbone_col << 1 | int(
                memory_fields[MemoryFields.VerticalInvert]
            )

        # show_amoeba_map(self.amoeba_map, retracts, moves, "Current Amoeba, Retracts, and Extends")

        return retracts, moves, info
