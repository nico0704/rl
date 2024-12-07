import matplotlib.pyplot as plt
from environment import TaxiEnvironment
from agent import QLearningAgent
from utils import (
    visualize_city,
    visualize_episode_with_imgs,
    create_city_graph
)


Q_TABLE_PATH = "q_tables/qt_10000000.txt"
ITERATIONS = 10


def run():
    city, positions = create_city_graph()
    visualize_city(city, positions)

    # initialize environment and agent
    env = TaxiEnvironment(city)
    agent = QLearningAgent(env, mode="run", q_table_path=Q_TABLE_PATH)
    
    # visu setup
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # run agent ITERATION times
    for _ in range(ITERATIONS):
        state = env.reset()
        done = False
        positions_trace = []
        while not done:
            action = agent.choose_action_deploy(state)
            next_state, reward, done = env.step(action)
            positions_trace.append(env.taxi_position)
            visualize_episode_with_imgs(ax, city, positions, env, positions_trace)
            plt.pause(0.5)
            state = next_state


if __name__ == "__main__":
    run()
