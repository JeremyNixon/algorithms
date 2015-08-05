import heapq
import sys

class Vertex:

	def __init__(self, node):
		self.id = node
		self.adjacent = {}
		self.distance = sys.maxint
		self.visited = False
		self.previous = None

	def add_neighbor(self, neighbor, weight = 0):
		self.adjacent[neighbor] = weight


	def set_distance(self, dist):
		self.distance = dist

	def get_distance(self):
		return self.distance

	def set_visited(self):
		self.visited = True

	def get_weight(self, neightbor):
		return self.adjacent[neighbor]


class Graph:
	# I bet that this is called whenever the class is initialized.
	def __init__(self):
		self.vert_dict = {}
		self.num_vertices = 0

	def __iter__(self):
		return iter(self.vert_dict.values())

	def get_vertex(self, n):
		if n in self.vert_dict:
			return self.vert_dict[n]
		else:
			return None

	def add_vertex(self, node):
		self.num_vertices += 1
		new_vertex = Vertex(node)
		self.vert_dict[node] = new_vertex
		return new_vertex

	def add_edge(self, frm, to, cost = 0):
		if frm not in self.vert_dict:
			self.add_vertex(frm)
		if to not in self.vert_dict:
			self.add_vertex(to)

		self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
		self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)



if __name__ == '__main__':

	g = Graph()
	g.add_vertex('a')
	g.add_vertex('b')
	g.add_vertex('c')

	g.add_edge('a', 'b', 7)
	g.add_edge('a', 'c', 9)
	g.add_edge('a', 'f', 14)
	g.add_edge('b', 'c', 10)
	g.add_edge('b', 'd', 15)
	g.add_edge('c', 'd', 11)
	g.add_edge('c', 'f', 2)
	g.add_edge('d', 'e', 6)
	g.add_edge('e', 'f', 9)

	aGraph = g
	start = g.get_vertex('a')

	print g.vert_dict
	print start

	# Set the distance for the start node to zero
	start.set_distance(0)

	unvisited_queue = [(v.get_distance(),v) for v in aGraph]
	heapq.heapify(unvisited_queue)

	while len(unvisited_queue) != 0:
		# pop the vertex with the smallest distance
		uv = heapq.heappop(unvisited_queue)
		current = uv[1]
		current.set_visited()

		# for next in v.adjacent:
		for next in current.adjacent:
			if next.visited:
				continue
			new_dist = current.get_distance() + current.get_weight(next)





