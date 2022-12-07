import logging
import math
import os
import pickle
from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import constants
from amoeba_state import AmoebaState

turn = 0


class PlayerParameters:
    formation_threshold: float

    def __init__(self, formation_threshold):
        self.formation_threshold = formation_threshold


default_params = PlayerParameters(0.9)

# ---------------------------------------------------------------------------- #
#                               Constants                                      #
# ---------------------------------------------------------------------------- #

CENTER_X = constants.map_dim // 2
CENTER_Y = constants.map_dim // 2

COMB_SEPARATION_DIST = 4
TEETH_GAP = 3

VERTICAL_SHIFT_PERIOD = 2
VERTICAL_SHIFT_LIST = (
    (
        [0 for i in range(VERTICAL_SHIFT_PERIOD)]
        + [1 for i in range(VERTICAL_SHIFT_PERIOD)]
    )
    * (round(np.ceil(100 / (VERTICAL_SHIFT_PERIOD * 2))))
)[:100]

FORMATION_THRESHOLD = 0.9

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
    plt.pcolormesh(map, edgecolors="k", linewidth=1)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.title(title)
    # plt.savefig(f"debug/{turn}.png")
    plt.show()


# ---------------------------------------------------------------------------- #
#                                Memory Bit Mask                               #
# ---------------------------------------------------------------------------- #


class MemoryFields(Enum):
    Initialized = 0
    Translating = 1


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


if __name__ == "__main__":
    memory = 0
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == False
    assert fields[MemoryFields.Translating] == False

    memory = change_memory_field(memory, MemoryFields.Initialized, True)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == True
    assert fields[MemoryFields.Translating] == False

    memory = change_memory_field(memory, MemoryFields.Translating, True)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == True
    assert fields[MemoryFields.Translating] == True

    memory = change_memory_field(memory, MemoryFields.Translating, False)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == True
    assert fields[MemoryFields.Translating] == False

    memory = change_memory_field(memory, MemoryFields.Initialized, False)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == False
    assert fields[MemoryFields.Translating] == False


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

    def add_cell(self, x, y) -> None:
        self.map[x % constants.map_dim, y % constants.map_dim] = 1

    def get_cell(self, x, y) -> int:
        return self.map[x % constants.map_dim, y % constants.map_dim]

    def merge_formation(self, formation_map: npt.NDArray):
        self.map = np.logical_or(self.map, formation_map)


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

        # Class accessible percept variables, written at the start of each turn
        self.current_size: int = None
        self.amoeba_map: npt.NDArray = None
        self.bacteria_cells: set[Tuple[int, int]] = None
        self.retractable_cells: List[Tuple[int, int]] = None
        self.extendable_cells: List[Tuple[int, int]] = None
        self.num_available_moves: int = None

    def generate_comb_formation(
        self,
        size: int,
        tooth_offset=0,
        center_x=CENTER_X,
        center_y=CENTER_Y,
        comb_idx=0,
    ) -> npt.NDArray:
        formation = Formation()
        comb_0_center_x = center_x

        if size < 2:
            return formation.map

        teeth_size = min((size // ((TEETH_GAP + 1) * 2 + 1)), 49)
        backbone_size = min((size - teeth_size) // 2, 99)
        cells_used = backbone_size * 2 + teeth_size

        # If we have hit our max size, form an additional comb and connect it via a bridge
        if backbone_size == 99 and comb_idx == 0:
            center_x_offset = np.absolute(center_x - CENTER_X)

            if size > cells_used * 2:
                # Offset each comb to the left and right of the map center
                comb_0_center_x = min(
                    max(CENTER_X - center_x_offset, COMB_SEPARATION_DIST), 50
                )
                comb_1_center_x = min(
                    max(
                        CENTER_X + center_x_offset + COMB_SEPARATION_DIST,
                        CENTER_X + COMB_SEPARATION_DIST + 1,
                    ),
                    100,
                )
            else:
                comb_1_center_x = center_x + COMB_SEPARATION_DIST

            # Generate the second comb
            second_comb = self.generate_comb_formation(
                size - cells_used,
                tooth_offset,
                comb_1_center_x % constants.map_dim,
                center_y,
                1,
            )
            formation.merge_formation(second_comb)

            # Bridge between the two combs
            for i in range(comb_0_center_x, comb_1_center_x):
                formation.add_cell(i, center_y)

        # Build first comb formation
        formation.add_cell(comb_0_center_x, center_y)
        formation.add_cell(comb_0_center_x - 1, center_y)
        for i in range(1, round((backbone_size - 1) / 2 + 0.1) + 1):
            # first layer of backbone
            formation.add_cell(comb_0_center_x, center_y + i)
            formation.add_cell(comb_0_center_x, center_y - i)
            # second layer of backbone
            formation.add_cell(comb_0_center_x - 1, center_y + i)
            formation.add_cell(comb_0_center_x - 1, center_y - i)
        for i in range(
            1,
            round(min((teeth_size * (TEETH_GAP + 1)) / 2, backbone_size / 2) + 0.1),
            TEETH_GAP + 1,
        ):
            formation.add_cell(comb_0_center_x + 1, center_y + tooth_offset + i)
            formation.add_cell(comb_0_center_x + 1, center_y + tooth_offset - i)

        # If we build a second comb, build up additional cells in the center
        if backbone_size == 99 and comb_idx == 0:
            cells_remaining = size - np.count_nonzero(formation.map)
            bridge_offset = 1
            while cells_remaining > 0 and bridge_offset < 99:
                for i in range(comb_0_center_x, comb_1_center_x):
                    offset = (
                        bridge_offset if bridge_offset <= 49 else 50 - bridge_offset
                    )
                    if formation.get_cell(i, center_y + offset) == 0:
                        formation.add_cell(i, center_y + offset)
                        cells_remaining -= 1
                        if cells_remaining <= 0:
                            break
                bridge_offset += 1

        # show_amoeba_map(formation.map)
        return formation.map

    def get_morph_moves(
        self, desired_amoeba: npt.NDArray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Function which takes a starting amoeba state and a desired amoeba state and generates a set of retracts and extends
        to morph the amoeba shape towards the desired shape.
        """

        current_points = map_to_coords(self.amoeba_map)
        desired_points = map_to_coords(desired_amoeba)

        potential_retracts = [
            p
            for p in list(set(current_points).difference(set(desired_points)))
            if p in self.retractable_cells
        ]
        potential_extends = [
            p
            for p in list(set(desired_points).difference(set(current_points)))
            if p in self.extendable_cells
        ]
        potential_extends.sort(key=lambda p: p[1])

        # show_amoeba_map(desired_amoeba, title="Desired Amoeba")
        # show_amoeba_map(self.amoeba_map, potential_retracts, potential_extends, title="Current Amoeba, Potential Retracts and Extends")

        # Loop through potential extends, searching for a matching retract
        retracts = []
        extends = []
        check_calls = 0
        for potential_extend in [p for p in potential_extends]:
            # Ensure we only move as much as possible given our current metabolism
            if len(extends) >= self.num_available_moves:
                break

            matching_retracts = list(potential_retracts)
            matching_retracts.sort(key=lambda p: math.dist(p, potential_extend))

            for i in range(len(matching_retracts)):
                retract = matching_retracts[i]
                # Matching retract found, add the extend and retract to our lists
                if self.check_move(retracts + [retract], extends + [potential_extend]):
                    check_calls += 1
                    retracts.append(retract)
                    potential_retracts.remove(retract)
                    extends.append(potential_extend)
                    potential_extends.remove(potential_extend)
                    break
                check_calls += 1
        # print(f"Check calls: {check_calls} / {self.current_size}")

        # If we have moves remaining, try and get closer to the desired formation
        # if len(extends) < self.num_available_moves and len(potential_retracts):
        #     desired_extends = [p for p in list(set(desired_points).difference(set(current_points))) if p not in self.extendable_cells]
        #     unused_extends = [p for p in self.extendable_cells if p not in extends]

        #     for potential_retract in [p for p in potential_retracts]:
        #         for desired_extend in desired_extends:
        #             curr_dist = math.dist(potential_retract, desired_extend)

        #             matching_extends = [p for p in unused_extends if self.check_move(retracts + [potential_retract], extends + [p])]
        #             matching_extends.sort(key=lambda p: math.dist(p, desired_extend))

        #             if len(matching_extends) and  math.dist(potential_retract, matching_extends[0]) < curr_dist:
        #                 # show_amoeba_map(self.amoeba_map, [potential_retract], [matching_extends[0]])
        #                 retracts.append(potential_retract)
        #                 potential_retracts.remove(potential_retract)
        #                 extends.append(matching_extends[0])
        #                 unused_extends.remove(matching_extends[0])
        #                 break

        # show_amoeba_map(self.amoeba_map, retracts, extends, title="Current Amoeba, Selected Retracts and Extends")
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
        global turn
        turn += 1

        self.store_current_percept(current_percept)

        retracts = []
        moves = []

        memory_fields = read_memory(info)
        if not memory_fields[MemoryFields.Initialized]:
            retracts, moves = self.get_morph_moves(
                self.generate_comb_formation(self.current_size, 0)
            )
            if len(moves) == 0:
                info = change_memory_field(info, MemoryFields.Initialized, True)
                info = (50 << 1) | info
                memory_fields = read_memory(info)

        if memory_fields[MemoryFields.Initialized]:
            # Extract backbone column from memory
            curr_backbone_col = info >> 1
            vertical_shift = VERTICAL_SHIFT_LIST[curr_backbone_col]
            next_comb = self.generate_comb_formation(
                self.current_size, vertical_shift, curr_backbone_col, CENTER_Y
            )
            # Check if current comb formation is filled
            comb_mask = self.amoeba_map[next_comb.nonzero()]
            settled = (
                sum(comb_mask) / len(comb_mask)
            ) > self.params.formation_threshold
            if not settled:
                retracts, moves = self.get_morph_moves(next_comb)

                # Actually, we have no more moves to make
                if len(moves) == 0:
                    settled = True

            if settled:
                # When we "settle" into the target backbone column, advance the backbone column by 1
                prev_backbone_col = curr_backbone_col
                new_backbone_col = (prev_backbone_col + 1) % 100
                vertical_shift = VERTICAL_SHIFT_LIST[new_backbone_col]
                next_comb = self.generate_comb_formation(
                    self.current_size, vertical_shift, prev_backbone_col, CENTER_Y
                )
                retracts, moves = self.get_morph_moves(next_comb)
                info = new_backbone_col << 1 | 1

        return retracts, moves, info
