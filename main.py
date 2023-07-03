import time
from game import Game


def main():
    game = Game(input("Your name: "))

    while True:
        winner = game.start()

        print(winner)
        time.sleep(3)


if __name__ == "__main__":
    main()
