import threading
import time
import numpy as np
import random


# Tic-Tac-Toe game environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.winner = None
        self.game_over = False

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.winner = None
        self.game_over = False

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, move):
        if not self.game_over and self.board[move[0], move[1]] == 0:
            self.board[move[0], move[1]] = self.current_player
            self.check_game_over()
            self.current_player = -self.current_player

    def check_game_over(self):
        for player in [-1, 1]:
            # Check rows
            if np.any(np.all(self.board == player, axis=1)):
                self.winner = player
                self.game_over = True
                return

            # Check columns
            if np.any(np.all(self.board == player, axis=0)):
                self.winner = player
                self.game_over = True
                return

            # Check diagonals
            if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
                self.winner = player
                self.game_over = True
                return

            # Check for a draw
            if np.all(self.board != 0):
                self.game_over = True
                return

    def print_board(self):
        symbols = {0: " ", 1: "X", -1: "O"}
        for row in self.board:
            print("|".join([symbols[s] for s in row]))
            print("-----")
        print()


# Reinforcement Learning Agent
class RLAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_q_value(self, state, action):
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros((3, 3))
        return self.q_table[state_str][action[0], action[1]]

    def update_q_value(self, state, action, value):
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros((3, 3))
        self.q_table[state_str][action[0], action[1]] = value

    def choose_action(self, state, valid_moves, episode):
        exploration_rate = self.epsilon / (episode + 1)

        if random.uniform(0, 1) < exploration_rate:
            return random.choice(valid_moves)
        else:
            q_values = np.array([self.get_q_value(state, move) for move in valid_moves])
            return valid_moves[np.argmax(q_values)]

    def train(self, episodes, num_threads=4):
        # Create a threading.Lock object for synchronization
        lock = threading.Lock()

        # Define a helper function for each thread
        def train_thread(thread_id):
            env = TicTacToe()
            for episode in range(thread_id, episodes, num_threads):
                env.reset()
                state = env.board.copy()

                while not env.game_over:
                    valid_moves = env.get_valid_moves()
                    action = self.choose_action(state, valid_moves, episode)

                    env.make_move(action)
                    next_state = env.board.copy()

                    if env.game_over:
                        reward = 1 if env.winner == 1 else -1
                    else:
                        reward = 0

                    lock.acquire()  # Acquire the lock before updating the Q-table
                    current_q = self.get_q_value(state, action)
                    next_max_q = np.max([self.get_q_value(next_state, move) for move in valid_moves])

                    new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
                    self.update_q_value(state, action, new_q)

                    lock.release()  # Release the lock after updating the Q-table

                    state = next_state

                if episode % 50 == 0:
                    print(episode)

        # Create and start the threads
        threads: list[threading.Thread] = []
        for i in range(num_threads):
            thread = threading.Thread(target=train_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        print("Training complete!")

    def play(self):
        env = TicTacToe()
        env.print_board()

        episode = 0  # Add episode counter

        while not env.game_over:
            if env.current_player == 1:
                # Agent's turn
                state = env.board.copy()
                valid_moves = env.get_valid_moves()
                action = self.choose_action(state, valid_moves, episode)  # Pass episode number
                env.make_move(action)

                print("Agent's move:")
                env.print_board()
            else:
                # Human's turn
                valid_moves = env.get_valid_moves()
                print("Valid moves:", valid_moves)
                row = int(input("Enter the row (0-2): "))
                col = int(input("Enter the column (0-2): "))
                action = (row, col)
                env.make_move(action)

                print("Your move:")
                env.print_board()

        if env.winner is None:
            print("It's a draw!")
        elif env.winner == 1:
            print("Agent wins!")
        else:
            print("You win!")


# Training the RL agent
agent = RLAgent()
agent.train(episodes=1000)

time.sleep(1)

# Playing against the agent
while True:
    agent.play()
