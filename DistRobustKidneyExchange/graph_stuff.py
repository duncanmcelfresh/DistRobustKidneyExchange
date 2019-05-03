import kidney_ndds
import kidney_digraph

import string
import random

VERB = False
# functions for graph flow algorithms
# (adapted from https://github.com/bigbighd604/Python/blob/master/graph/Ford-Fulkerson.py)
class Edge(object):
  def __init__(self, u, v, w):
    self.source = u
    self.target = v
    self.capacity = w
    self.residual = w

  def __repr__(self):
    return "%s->%s:%s" % (self.source, self.target, self.capacity)


class FlowNetwork(object):
  def  __init__(self):
    self.adj = {}
    self.flow = {}
    self.residuals = {}
    self.names = []

  def AddVertex(self, name=None):
    if name is None:
        name = self.UnusedName()

    if name in self.adj:
        raise Warning("vertex exists with name : %s" % name)
    else:
        self.adj[name] = []
        self.names.append(name)

  def GetEdges(self, v):
    return self.adj[v]

  def AddEdge(self, u, v, w = 0):
    if u == v:
      raise ValueError("u == v")
    edge = Edge(u, v, w)
    redge = Edge(v, u, 0)
    edge.redge = redge
    redge.redge = edge
    self.adj[u].append(edge)
    self.adj[v].append(redge)
    # Intialize all flows to zero
    self.flow[edge] = 0
    self.flow[redge] = 0

  def FindPath(self, source, target, path):
    if source == target:
      return path
    for edge in self.GetEdges(source):
      residual = edge.capacity - self.flow[edge]
      if residual > 0 and not (edge, residual) in path:
        result = self.FindPath(edge.target, target, path + [(edge, residual)])
        if result != None:
          return result

  def MaxFlow(self, source, target):
    # reset flows to 0:
    for key in self.flow:
        self.flow[key] = 0
        self.residuals[key] = 0
    path = self.FindPath(source, target, [])
    if VERB: print 'path after enter MaxFlow: %s' % path
    for key in self.flow:
      if VERB: print '%s:%s' % (key,self.flow[key])
      if VERB: print '-' * 20
    while path != None:
      flow = min(res for edge, res in path)
      for edge, res in path:
        self.flow[edge] += flow
        self.flow[edge.redge] -= flow
      for key in self.flow:
          if VERB: print '%s:%s' % (key,self.flow[key])
      path = self.FindPath(source, target, [])
      if VERB: print 'path inside of while loop: %s' % path
    for key in self.flow:
        if VERB: print '%s:%s' % (key,self.flow[key])
    return sum(self.flow[edge] for edge in self.GetEdges(source))

  def UnusedName(self,N=5):
      for i in range(1000):
          rand_name = RandomString(N)
          if rand_name not in self.adj:
              return rand_name
              break
      raise Warning("Houston, we have a problem.")

def RandomString(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

if __name__ == "__main__":
  g = FlowNetwork()
  for _ in range(10): g.AddVertex()

  for _ in range(10):
      v1, v2 = random.sample(g.names, 2)
      wt = random.random()
      g.AddEdge(v1, v2, wt)
  # map(g.AddVertex, ['s', 'o', 'p', 'q', 'r', 't','o'])
  # g.AddEdge('s', 'o', 5)
  # g.AddEdge('s', 'p', 3)
  # g.AddEdge('o', 'p', 2)
  # g.AddEdge('o', 'q', 3)
  # g.AddEdge('p', 'r', 4)
  # g.AddEdge('r', 't', 3)
  # g.AddEdge('q', 'r', 4)
  # g.AddEdge('q', 't', 2)

  print g.MaxFlow(g.names[0], g.names[1])