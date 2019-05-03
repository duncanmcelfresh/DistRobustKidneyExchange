from gurobipy import *

TIME_LIM = 18000 #18,000s = 5 hours

# MIP constants
EPS_MIP = 1e-3
EPS_SMALL = EPS_MIP
M = 1e3


def create_mip_model(time_lim=TIME_LIM, verbose=True, mipgap=None):
    """Create a Gurobi MIP model."""

    m = Model("mip")
    if not verbose:
        m.params.outputflag = 0

    if time_lim is not None:
        m.params.timelimit = time_lim
    if mipgap is not None:
        assert mipgap >= 0
        m.params.MIPGap = mipgap
    return m


def optimize(model):
    '''optimize a Gurobi model, and check status'''

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        raise Warning("model is infeasible")
        m.computeIIS()
        m.write('model.mps')
        m.write('model.rlp')
        m.write('model.ilp')
    elif model.status == GRB.UNBOUNDED:
        raise Warning("model is unbounded")

    if model.status == GRB.TIME_LIMIT:
        raise Warning("time limit reached")
