import matplotlib.pyplot as plt
from dolfin import *

def load_ellipsoid_data():
    """Returns 4-tuple Mesh, MeshFunction, dict of marking numbers and dict of fiber functions.
    The MeshFunction numbers domains according to the numbering dict."""
    comm = mpi_comm_world()
    hdf5file = dolfin.HDF5File(dolfin.mpi_comm_world(), "simple_ellipsoid.h5", "r")

    mesh = Mesh(comm)
    hdf5file.read(mesh, "/geometry/mesh", True)

    mf = MeshFunction("size_t", mesh, 2, mesh.domains()) # 2 = dimension of entities (facets)
    hdf5file.read(mf, "/geometry/mesh/meshfunction_2")

    # hack to make function look nicer in Paraview
    # as-is interior facets have a huge int which makes other
    # values look the same, but irrelevant for BCs
    big_num = max(mf.array())
    mf.array()[mf.array() == big_num] = 0
    
    File("domain_markers.pvd") << mf # may be opened with paraview

    numbering = {
        "BASE": 10,
        "ENDO": 30,
        "EPI": 40
    }


    # load fibers

    try:
        family, order = hdf5file.attributes("/microstructure")["space"].split("_")
        order = int(order)
    except:
        family, order = "Quadrature", 4
    elm = dolfin.VectorElement(family=family,
                                       cell=mesh.ufl_cell(),
                                       degree=int(order),
                                       quad_scheme="default")
    V = dolfin.FunctionSpace(mesh, elm)
    
    fibers = {}
    fiber_names = hdf5file.attributes("/microstructure")["names"].split(":")
    for name in fiber_names:
        fiberfunc = Function(V, name=name)
        hdf5file.read(fiberfunc, "/microstructure/{}".format(name))
        fibers[name] = fiberfunc


    for name in fibers:
        File(name+'.xml') << fibers[name]
        #fibers[name] >> filename
    # should contain:
    # 'cross_sheet_fepi-60_fendo60_sepi0_sendo0',
    # 'fiber_epi-60_endo60',
    # 'sheet_epi0_endo0'
    

    return mesh, mf, numbering, fibers
                 



# Adjust log level
set_log_level(PROGRESS)
parameters["form_compiler"]["cpp_optimize"] = True

mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()

V = VectorFunctionSpace(mesh,'P',1)

# Redefine boundary measure
ds = Measure('ds',domain=mesh,subdomain_data=boundary_markers)

clamp = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, clamp, boundary_markers, numbering["BASE"])
bcs = [bc]

# Define solution u and test function v
u = Function(V)
v = TestFunction(V)

# Define constants
mu    = 0.0023 * 1e3 # N  / m^2 Lame parameter mu for PVC plastic
lmbda = 0.0105 * 1e3 # N  / m^2 Lame parameter lambda for PVC plastic
mu    = Constant(mu)
lmbda = Constant(lmbda)

# Define strain measures
I = Identity(3) # the identity matrix
F = I + grad(u) # the deformation gradient
C = F.T*F       # the right Cauchy-Green tensor
E = 0.5*(C - I) # the Green-Lagrange strain tensor

# Define strain energy density
E = variable(E)
W = lmbda/2*(tr(E)**2) + mu*tr(E*E)


S = diff(W, E) # the second Piola-Kirchoff stress tensor
P = F*S        # the first Piola-Kirchoff stress tensor
p_endo = 1.0e2

# Define nonlinear problem
N = FacetNormal(mesh)
Gext = p_endo * inner(v, cofac(F)*N) * ds(numbering["ENDO"]) #_bottom
R = inner(P,grad(v))*dx + Gext #dot(T,v)*ds(2)
#J = derivative(R,u,du)

solve(R==0,u,bcs)

# Save solution to file
file = File("displacement.pvd")
file << u


