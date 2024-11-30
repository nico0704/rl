import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def save_q_table(agent):
    with open("qtable.txt", "w") as file:
        for entry in agent.q_table:
            formatted_values = {
                action: f"{value:.2f}"
                for action, value in agent.q_table[entry].items()
            }
            file.write(f"{entry}: {formatted_values}\n")


def visualize_episode(env, agent, city, positions):
    state = env.reset()
    env.taxi_position = 12
    positions_trace = [env.taxi_position]
    passenger_start = env.passenger_start
    passenger_destination = env.passenger_destination

    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)
        positions_trace.append(env.taxi_position)
        state = next_state

    print(f"path: {positions_trace}")

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(num):
        ax.clear()
        nx.draw(
            city,
            pos=positions,
            with_labels=True,
            node_color="lightblue",
            node_size=800,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            city,
            pos=positions,
            nodelist=[passenger_start],
            node_color="green",
            node_size=1000,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            city,
            pos=positions,
            nodelist=[passenger_destination],
            node_color="red",
            node_size=1000,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            city,
            pos=positions,
            nodelist=[positions_trace[num]],
            node_color="yellow",
            node_size=1000,
            ax=ax,
        )
        edge_labels = nx.get_edge_attributes(city, "weight")
        nx.draw_networkx_edge_labels(
            city, pos=positions, edge_labels=edge_labels
        )
        # legend for colors
        # labels = ["Passenger-Start", "Passenger-Destination", "Taxi"]
        # colors = ["green", "red", "yellow"]
        # for color in colors:
        #     plt.scatter([], [], color=color, label=labels[colors.index(color)])
        # plt.legend(loc="upper left")

        def add_image(ax, img_path, position):
            img = mpimg.imread(img_path)
            imagebox = OffsetImage(img, zoom=0.3)  # Adjust `zoom` as needed
            ab = AnnotationBbox(imagebox, position, frameon=False)
            ax.add_artist(ab)

        add_image(ax, "assets/taxi.png", positions[positions_trace[num]])
        add_image(ax, "assets/passenger.png", positions[passenger_start])
        add_image(ax, "assets/flag.png", positions[passenger_destination])

    time.sleep(1)
    ani = animation.FuncAnimation(
        fig, update, frames=len(positions_trace), interval=1000, repeat=False
    )
    plt.show()


def visualize_heatmap(city, edge_usage, positions):
    # Create Heatmap from edge usage
    edge_colors = []
    max_usage = max(edge_usage.values()) if edge_usage.values() else 1

    # Used colors for the Heatmap
    colors = ["#FFE5B4", "#FF8C42", "#FF0000"]
    colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)

    for edge in city.edges:
        usage = edge_usage[edge]
        normalized_usage = usage / max_usage  # Normalise [0,1]
        color = colormap(normalized_usage)  # Color based on Normalisation
        edge_colors.append(color)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the city
    nx.draw(
        city,
        pos=positions,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        edge_color=edge_colors,
        width=8,
        ax=ax,
    )

    # Create color bar
    sm = plt.cm.ScalarMappable(
        cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max_usage)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Edge usage", rotation=270, labelpad=20)

    plt.title("Heatmap for edge usage")
    plt.show()
