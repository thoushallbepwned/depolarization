import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as opn
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution

# Network topology
#g = nx.erdos_renyi_graph(1000, 0.1)
#nx.set_node_attributes(g, ('opinion'+1)/2, 'opinion')

g = nx.read_gml("data/graph_structures/opinion_seeded/normal/graph_0.gml")

# Model selection
model = opn.AlgorithmicBiasModel(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", 0.25)
config.add_model_parameter("gamma", 0)
model.set_initial_status(config)

# Simulation execution
iterations = model.iteration_bunch(20)

viz = OpinionEvolution(model, iterations)
viz.plot("opinion_ev.pdf")