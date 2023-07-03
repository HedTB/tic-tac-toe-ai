import random
import time
from typing import Literal
import numpy
from pydantic import BaseModel


WINNING_COMBINATIONS = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 5, 9], [3, 5, 7]]
PIECES: list[Literal["X"] | Literal["O"]] = ["X", "O"]

Piece = Literal["X"] | Literal["O"]


def state_to_num(state: list[int]):
    return (
        state[0]
        + 3 * state[1]
        + 9 * state[2]
        + 27 * state[3]
        + 81 * state[4]
        + 243 * state[5]
        + 729 * state[6]
        + 2187 * state[7]
        + 6561 * state[8]
    )


def num_to_state(num: int):
    i = num // (3**8)
    h = (num - i * (3**8)) // (3**7)
    g = (num - i * (3**8) - h * (3**7)) // (3**6)
    f = (num - i * (3**8) - h * (3**7) - g * (3**6)) // (3**5)
    e = (num - i * (3**8) - h * (3**7) - g * (3**6) - f * (3**5)) // (3**4)
    d = (num - i * (3**8) - h * (3**7) - g * (3**6) - f * (3**5) - e * (3**4)) // (3**3)
    c = (num - i * (3**8) - h * (3**7) - g * (3**6) - f * (3**5) - e * (3**4) - d * (3**3)) // (3**2)
    b = (
        num - i * (3**8) - h * (3**7) - g * (3**6) - f * (3**5) - e * (3**4) - d * (3**3) - c * (3**2)
    ) // (3**1)
    a = (
        num
        - i * (3**8)
        - h * (3**7)
        - g * (3**6)
        - f * (3**5)
        - e * (3**4)
        - d * (3**3)
        - c * (3**2)
        - b * (3**1)
    ) // (3**0)

    return [a, b, c, d, e, f, g, h, i]


class Player(BaseModel):
    name: str
    piece: Piece
    positions: list[int] = []

    is_ai: bool

    epsilon: float = 1.0
    values: dict[int, float] = {0: 0}


class Tile(BaseModel):
    occupied: bool = False
    player: Player | None = None

    def occupy(self, player: Player):
        self.occupied = True
        self.player = player

    def __str__(self) -> str:
        return self.player.piece if self.player else " "


class Game:
    def __init__(self, human_name: str) -> None:
        x_name = human_name if random.randint(0, 1) == 0 else "AI"
        o_name = "AI" if x_name == human_name else human_name

        self.tiles = [Tile() for _ in range(9)]
        self.players = [
            # Player(name=x_name, piece="X", is_ai=x_name == "AI"),
            # Player(name=o_name, piece="O", is_ai=o_name == "AI"),
            Player(name=x_name, piece="X", is_ai=True),
            Player(name=o_name, piece="O", is_ai=True),
        ]

        print(self.tiles)
        print(self.players)

        self.current_player = self.players[0]

    @property
    def tiles_int_list(self):
        int_list: list[int] = []

        for tile in self.tiles:
            int_list.append(0 if not tile.player else (1 if tile.player.piece == "X" else 2))

        return int_list

    def update_values(self, player: Player, state_history: list[int], win=False):
        player.values[state_history[-1]] = -1

        if win:
            player.values[state_history[-1]] = 1

        for state in state_history:
            if state not in player.values:
                player.values[state] = 0

        for i in range(len(state_history) - 1, 0, -1):
            player.values[state_history[i - 1]] += 0.1 * (
                player.values[state_history[i]] - player.values[state_history[i - 1]]
            )

    def get_available_positions(self):
        return [index for index, tile in enumerate(self.tiles, start=1) if not tile.occupied]

    def get_action(self):
        if not self.current_player.is_ai:
            try:
                return int(input("Your move: "))
            except ValueError:
                print("Invalid input!")
                return None

        marker = self.players.index(self.current_player) + 1
        epsilon = self.current_player.epsilon

        possible_next_states: dict[int, int] = {}
        top_value = -1

        for i in range(len(self.tiles_int_list)):
            if self.tiles_int_list[i] == 0:
                copy = numpy.copy(self.tiles_int_list)
                copy[i] = marker

                possible_next_states[i] = state_to_num(copy)

        if numpy.random.rand() < epsilon:
            player = self.players[marker - 1]

            if player.epsilon > 0.05:
                player.epsilon -= 0.001

            return random.sample(possible_next_states.keys(), 1)[0]
        else:
            i = 0

            for state in possible_next_states.values():
                try:
                    player = self.players[marker - 1]

                    if player.values(state) > top_value:
                        top_value = player.values(state)
                        action = list(possible_next_states.keys())[i]
                except Exception:
                    pass

                i += 1

            player = self.players[marker - 1]

            if player.epsilon > 0.05:
                player.epsilon -= 0.001

            try:
                return action
            except Exception:
                return random.sample(possible_next_states.keys(), 1)[0]

    def start(self) -> tuple[Piece, str] | None:
        state_history = []

        self.output()

        while True:
            position = self.get_action()

            if not position:
                time.sleep(0.5)
                continue

            try:
                self.make_move(position)
            except Exception as error:
                print(error.args[0])
                time.sleep(0.5)
                continue

            state_history.append(state_to_num(self.tiles_int_list))
            result = self.check_end()

            if result == 1:
                print("Win!")
                return self.current_player.piece, self.current_player.name
            elif result == 2:
                print("Draw!")
                return None
            else:
                time.sleep(0.5)
                self.output()

    def output(self):
        print("\n")
        print("\t     |     |")
        print("\t  {}  |  {}  |  {}".format(self.tiles[0], self.tiles[1], self.tiles[2]))
        print("\t_____|_____|_____")

        print("\t     |     |")
        print("\t  {}  |  {}  |  {}".format(self.tiles[3], self.tiles[4], self.tiles[5]))
        print("\t_____|_____|_____")

        print("\t     |     |")

        print("\t  {}  |  {}  |  {}".format(self.tiles[6], self.tiles[7], self.tiles[8]))
        print("\t     |     |")
        print("\n")

    def make_move(self, position: int):
        if position < 1 or position > 9:
            print(position, repr(position), repr(str(position)))
            raise Exception("Invalid position")

        print(position)

        tile = self.tiles[position - 1]
        print(tile)

        if tile.occupied:
            raise Exception("Occupied position")

        tile.occupy(self.current_player)

        self.current_player.positions.append(position)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]

    def check_win(self):
        for combination in WINNING_COMBINATIONS:
            if all(position in self.current_player.positions for position in combination):
                return True

        return False

    def check_draw(self):
        if len(self.players[0].positions) + len(self.players[1].positions) == 9:
            return True

        return False

    def check_end(self):
        if self.check_win():
            return 1
        elif self.check_draw():
            return 2
        else:
            return False
