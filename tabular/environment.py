import random


#### environment ####
class TaxiEnvironment:
    def __init__(self, city_graph):
        self.city = city_graph
        self.edge_usage = {edge: 0 for edge in self.city.edges}  # Track edge usage for heatmap
        self.reset()

    def reset(self):
        self.taxi_position = random.choice(list(self.city.nodes))
        self.passenger_status = 'no_passenger'
        self.passenger_start = random.choice(list(self.city.nodes))
        self.passenger_destination = random.choice(list(self.city.nodes))
        return (self.taxi_position, self.passenger_status, self.passenger_start, self.passenger_destination)

    def step(self, action):
        self.edge_usage[tuple(sorted([self.taxi_position, action]))] += 1
        edge_weight = self.city[self.taxi_position][action]['weight']
        self.taxi_position = action

        reward = -edge_weight  # movement cost
        done = False

        if self.passenger_status == 'no_passenger' and self.taxi_position == self.passenger_start:
            self.passenger_status = 'has_passenger'
            reward += 10
        elif self.passenger_status == 'has_passenger' and self.taxi_position == self.passenger_destination:
            reward += 20
            done = True

        return (self.taxi_position, self.passenger_status, self.passenger_start, self.passenger_destination), reward, done