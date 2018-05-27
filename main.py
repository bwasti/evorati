import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import math

fig = plt.gcf()
fig.show()
fig.canvas.draw()


class Agent(object):

    def __init__(self, bank, init_policy=2.5):
        self.policy = random.uniform(0, init_policy * 2)
        self.bank = bank

    def declared_price(self, intrinsic_cost):
        self.cost = intrinsic_cost
        self.price = max(self.policy * self.cost, 0)
        return self.price

    def pay(self, payment):
        self.bank -= self.cost
        self.bank += payment

    def clone(self, mutation):
        cloned = Agent(self.bank)
        cloned.policy = self.policy + random.uniform(-mutation, mutation)
        return cloned

    def __repr__(self):
        return "bank: {:10.4f}, policy: {:10.4f}x cost".format(self.bank, self.policy)


def get_edge_list(path):
    edge_list = []
    prev_p = path[0]
    for p in path[1:]:
        edge_list.append((prev_p, p))
        prev_p = p
    return edge_list


#### Game parameters ####
# this param slows stuff down a bit
use_erdos_renyi_graph = False  # else complete graph
num_nodes = 20
# 2 times the likely connected probability
erg_prob = 2 * 2 * math.log(num_nodes) / num_nodes

#### Runtime parameters ####
iters = 2500
batch_size = 400
batch_size_decay = 0
min_batch_size = 100

#### Agent behavior parameters ####
bankroll = 100
init_policy = 2.0  # initial average policy
num_agents = 400

#### Evolutionary parameters ####
# each batch we clone top_k
# insert add_k with avg wealth
# and remove bottom_k
top_k = 15
add_k = 5
bottom_k = 20
assert top_k + add_k == bottom_k


# Create a set of agents
agents = set()

for _ in range(num_agents):
    agents.add(Agent(bankroll, init_policy))

#### Plotting things ####
policies = []
avg_policies = []
top_policies = []

for iteration in range(iters * batch_size):
    # Create a graph with random weights (there are a couple ways to do this)
    source = 0
    sink = 4
    G = None
    if use_erdos_renyi_graph:
        G = nx.gnp_random_graph(num_nodes, erg_prob, directed=True)
        try:
            # Sanity check that the source and sink are connected
            # in two possible ways
            nodes = nx.shortest_path(G, source=source, target=sink)
            H = G.copy()
            for edge in get_edge_list(nodes):
                H.remove_edge(*edge)
            nx.shortest_path(H, source=source, target=sink)
        except:
            continue
    else:
        G = nx.complete_graph(num_nodes, nx.DiGraph())

    if len(G.edges) > len(agents):
        print("Not enough players to play the game")
        break

    for n, nbrdict in G.adjacency():
        for nbr, eattr in nbrdict.items():
            eattr["cost"] = random.randint(1, 10)

    # Give an edge to a sample of agents (not all are going to be playing)
    players = random.sample(agents, len(G.edges))
    edge_map = dict()  # edge_map[edge] --> player
    for player, edge in zip(players, G.edges):
        # Allow each agent to decide a price for their edge
        cost = G.edges[edge]["cost"]
        G.edges[edge]["price"] = player.declared_price(cost)
        edge_map[edge] = player

    # Calculate the shortest path and it's price
    path = nx.shortest_path(G, source=source, target=sink, weight="price")
    edge_list = get_edge_list(path)
    price = 0
    for edge in edge_list:
        price += G.edges[edge]["price"]

    # Pay each player on the shortest path
    for edge in edge_list:
        agent_price = G.edges[edge]["price"]
        saved_data = G.edges[edge]
        saved_edge = edge

        # Calculate externality of the player
        G.remove_edge(*edge)
        new_path = nx.shortest_path(G, source=source, target=sink, weight="price")
        new_edge_list = get_edge_list(new_path)
        new_price = 0
        for new_edge in new_edge_list:
            new_price += G.edges[new_edge]["price"]
        alt_price = price - agent_price
        G.add_edge(*saved_edge, **saved_data)

        # Play the player their externality
        edge_map[edge].pay(new_price - alt_price)

    # Prune poor agents and clone wealthy agents each batch
    if iteration % batch_size == 0:
        batch_size -= batch_size_decay
        batch_size = max(batch_size, min_batch_size)
        sorted_agents = sorted(agents, key=lambda x: x.bank)
        top_k_agents = sorted_agents[-top_k:]
        for agent in top_k_agents:
            agents.add(agent.clone(0.01))
        for agent in sorted_agents[:bottom_k]:
            agents.remove(agent)

        avg_wealth = sum([a.bank for a in agents]) / len(agents)
        avg_policy = sum([a.policy for a in agents]) / len(agents)

        top_wealth = sum([a.bank for a in top_k_agents]) / top_k
        top_policy = sum([a.policy for a in top_k_agents]) / top_k

        for _ in range(add_k):
            agents.add(Agent(avg_wealth, init_policy))

        print(
            "Top 1 \tpolicy: {:10.4f} \tvalue: {:10.4f}".format(
                sorted_agents[-1].policy, sorted_agents[-1].bank
            )
        )

        print(
            "Top {} \tpolicy: {:10.4f} \tvalue: {:10.4f}".format(
                top_k, top_policy, top_wealth
            )
        )

        print("Avg \tpolicy: {:10.4f} \tvalue: {:10.4f}".format(avg_policy, avg_wealth))
        print()

        policies.append(sorted_agents[-1].policy)
        avg_policies.append(avg_policy)
        top_policies.append(top_policy)

        plt.plot(range(len(policies)), policies, "C1", label="Winning")
        plt.plot(range(len(policies)), top_policies, "C2", label="Top " + str(top_k))
        plt.plot(range(len(policies)), avg_policies, "C3", label="Average")
        plt.plot(range(len(policies)), np.ones(len(policies)), "C4")
        if iteration == 0:
            plt.legend()

        plt.xlim([0, len(policies)])
        plt.ylim([0, 2])
        fig.canvas.draw()
