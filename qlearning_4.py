import pickle
import time

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from PIL import Image

style.use("ggplot")

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

EPS_DECAY = 0.9998
STEPS_PER_EPISODE = 200

LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Enum
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
NUM_BLOB_TYPES = 3

ACTION_UP_LEFT = 0
ACTION_DOWN_RIGHT = 1
ACTION_UP_RIGHT = 2
ACTION_DOWN_LEFT = 3
NUM_ACTIONS = 4

# colors defined in BGR format. why?
d = {
    PLAYER_N: (255, 175, 0),  # Orangish
    FOOD_N: (0, 255, 0),  # Green
    ENEMY_N: (0, 0, 255),  # Red
}


class Blob:
    def __init__(self):
        # Ideally, blobs would not be able to spawn on existing blobs.
        # We don't handle that.
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice: int):
        if choice == ACTION_UP_LEFT:
            self.move(x=1, y=1)
        elif choice == ACTION_DOWN_RIGHT:
            self.move(x=-1, y=-1)
        elif choice == ACTION_UP_RIGHT:
            self.move(x=-1, y=1)
        elif choice == ACTION_DOWN_LEFT:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)  # -1, 0, or 1
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)  # -1, 0, or 1
        else:
            self.y += y

        self.x = min(SIZE - 1, max(0, self.x))
        self.y = min(SIZE - 1, max(0, self.y))


@click.command()
@click.option("--start-q-table", help="filename of saved q-table", default=None)
@click.option("--epsilon", default=0.9)
@click.option("--show-every", default=3000)
@click.option("--enable-movement", default=False, is_flag=True)
def cli(start_q_table, epsilon, show_every, enable_movement):
    """"""

    if start_q_table is None:
        q_table = {}
        for x1 in range(-SIZE + 1, SIZE):
            for y1 in range(-SIZE + 1, SIZE):
                for x2 in range(-SIZE + 1, SIZE):
                    for y2 in range(-SIZE + 1, SIZE):
                        q_table[((x1, y1), (x2, y2))] = [
                            np.random.uniform(-5, 0) for i in range(NUM_ACTIONS)
                        ]
    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    episode_rewards = []
    for episode in range(HM_EPISODES):
        player = Blob()
        food = Blob()
        enemy = Blob()

        if episode % show_every == 0:
            print(f"on # {episode}, epsilon: {epsilon}")
            print(f"{show_every} ep mean {np.mean(episode_rewards[-show_every:])}")
            show = True
        else:
            show = False

        episode_reward = 0
        for i in range(STEPS_PER_EPISODE):
            obs = (player - food, player - enemy)

            # Take the action
            if np.random.random() > epsilon:
                action = np.argmax(q_table[obs])
            else:
                action = np.random.randint(0, NUM_ACTIONS)

            player.action(action)

            # Move the other objects
            if enable_movement:
                enemy.move()
                food.move()

            # Give the reward
            if player.x == enemy.x and player.y == enemy.y:
                reward = -ENEMY_PENALTY
            elif player.x == food.x and player.y == food.y:
                reward = FOOD_REWARD
            else:
                reward = -MOVE_PENALTY

            new_obs = (player - food, player - enemy)
            max_future_q = np.max(q_table[new_obs])
            current_q = q_table[obs][action]

            if reward == FOOD_REWARD:
                new_q = FOOD_REWARD
            elif reward == -ENEMY_PENALTY:
                new_q = -ENEMY_PENALTY
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                    reward + DISCOUNT * max_future_q
                )

            q_table[obs][action] = new_q

            if show:
                env = np.zeros((SIZE, SIZE, NUM_BLOB_TYPES), dtype=np.uint8)
                env[food.y][food.x] = d[FOOD_N]
                env[player.y][player.x] = d[PLAYER_N]
                env[enemy.y][enemy.x] = d[ENEMY_N]

                img = Image.fromarray(env, "RGB")
                img = img.resize((300, 300))
                cv2.imshow("", np.array(img))
                if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                    # Simulation ended
                    if cv2.waitKey(500) & 0xFF == ord("q"):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            episode_reward += reward
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                break
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

    moving_avg = np.convolve(
        episode_rewards, np.ones((show_every,)) / show_every, mode="valid"
    )

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"reward {show_every}ma")
    plt.xlabel("episode #")
    plt.show()

    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)


if __name__ == "__main__":
    cli()
