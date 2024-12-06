import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os


def plot_rewards(total_rewards, cumulative_rewards):
    os.makedirs("plots", exist_ok=True)
    # Total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label="Total Reward per Episode", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("plots/total_reward_plot.png")

    # Cumulative rewards
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_rewards, label="Cumulative Reward", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Episodes")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("plots/cumulative_reward_plot.png")


def save_q_table(agent):
    with open("qtable.txt", "w") as file:
        for entry in agent.q_table:
            formatted_values = {
                action: f"{value:.2f}"
                for action, value in agent.q_table[entry].items()
            }
            file.write(f"{entry}: {formatted_values}\n")
            

def visualize_city(city, positions):
    edge_labels = nx.get_edge_attributes(city, 'weight')
    plt.figure(figsize=(8, 8))
    nx.draw(city, pos=positions, with_labels=True, node_color='lightblue', node_size=800)
    nx.draw_networkx_edge_labels(city, pos=positions, edge_labels=edge_labels)
    plt.title("City Graph")
    plt.show()
    
    
def visualize_episode(ax, city, positions, env, positions_trace):
    ax.clear()
    nx.draw(
        city, pos=positions, with_labels=True,
        node_color="lightblue", node_size=800, ax=ax
    )
    passenger_color = "lightgreen" if env.passenger_status == "has_passenger" else "green"
    nx.draw_networkx_nodes(
        city, pos=positions, nodelist=[env.passenger_start],
        node_color=passenger_color, node_size=1000, ax=ax
    )
    nx.draw_networkx_nodes(
        city, pos=positions, nodelist=[env.passenger_destination],
        node_color="red", node_size=1000, ax=ax
    )
    path_edges = [
        (positions_trace[i], positions_trace[i + 1])
        for i in range(len(positions_trace) - 1)
    ]
    nx.draw_networkx_edges(
        city, pos=positions, edgelist=path_edges,
        edge_color="yellow", width=2.5, ax=ax
    )
    nx.draw_networkx_nodes(
        city, pos=positions, nodelist=[env.taxi_position],
        node_color="yellow", node_size=1000, ax=ax
    )
    edge_labels = nx.get_edge_attributes(city, "weight")
    nx.draw_networkx_edge_labels(city, pos=positions, edge_labels=edge_labels)
    labels = ["Passenger is waiting", "Passenger is picked up", "Destination", "Taxi"]
    colors = ["green", "lightgreen", "red", "yellow"]
    for color in colors:
        plt.scatter([], [], color=color, label=labels[colors.index(color)])
    plt.legend(loc="upper left")


def visualize_episode_with_imgs(ax, city, positions, env, positions_trace):
    ax.clear()
    nx.draw(
        city, pos=positions, with_labels=True,
        node_color="lightblue", node_size=800, ax=ax
    )
    passenger_color = "lightgreen" if env.passenger_status == "has_passenger" else "green"
    nx.draw_networkx_nodes(
        city, pos=positions, nodelist=[env.passenger_start],
        node_color=passenger_color, node_size=1000, ax=ax
    )
    nx.draw_networkx_nodes(
        city, pos=positions, nodelist=[env.passenger_destination],
        node_color="red", node_size=1000, ax=ax
    )
    path_edges = [
        (positions_trace[i], positions_trace[i + 1])
        for i in range(len(positions_trace) - 1)
    ]
    nx.draw_networkx_edges(
        city, pos=positions, edgelist=path_edges,
        edge_color="yellow", width=2.5, ax=ax
    )
    nx.draw_networkx_nodes(
        city, pos=positions, nodelist=[env.taxi_position],
        node_color="yellow", node_size=1000, ax=ax
    )
    edge_labels = nx.get_edge_attributes(city, "weight")
    nx.draw_networkx_edge_labels(city, pos=positions, edge_labels=edge_labels)

    def add_image(ax, img_path, position):
        img = mpimg.imread(img_path)
        imagebox = OffsetImage(img, zoom=0.25)  # Adjust `zoom` as needed
        ab = AnnotationBbox(imagebox, position, frameon=False)
        ax.add_artist(ab)

    add_image(ax, "assets/flag.png", positions[env.passenger_destination])
    if env.passenger_status == "has_passenger":
        add_image(ax, "assets/passenger.png", positions[env.taxi_position])
    else: add_image(ax, "assets/passenger.png", positions[env.passenger_start])
    add_image(ax, "assets/taxi.png", positions[env.taxi_position])


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
    plt.savefig("plots/heatmap.png")
    plt.show()
