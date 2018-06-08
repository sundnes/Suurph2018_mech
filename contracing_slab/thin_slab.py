import matplotlib.pyplot as plt
import numpy as np
from guccionematerial import *
from fenics import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4
#parameters["allow_extrapolation"] = True


L = 10
W = 1

mesh = BoxMesh(Point(0,0,0), Point(L,W,W),40,4,4)

V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = L)

boundary_markers = MeshFunction("size_t", mesh,mesh.topology().dim() - 1)
boundary_markers.set_all(0)
left.mark(boundary_markers, 1)
right.mark(boundary_markers, 2)

# Redefine boundary measure
ds = Measure('ds',domain=mesh,subdomain_data=boundary_markers)

#print ds(2)

# Define Dirichlet boundary (x = 0 or x = 1)
clamp = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, clamp, left)
bcs = [bc]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
T  = Constant((0.0,  0.0, 0.001))  # Traction force on the boundary


f0 = as_vector([ 1.0, 0.0, 0.0 ])
s0 = as_vector([ 0.0, 1.0, 0.0 ])
n0 = as_vector([ 0.0, 0.0, 1.0 ])
        

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
F = variable(F)

#define the strain energy
material = GuccioneMaterial(e1=f0, e2=s0, e3=n0,
                               C=2.0, bf=8, bt=2, bfs=4,kappa=1.0e4,Tactive=0.0)
psi = material.strain_energy(F)

#first Piola-Kirchoff stress
P = diff(psi,F)

p_right = Constant(-1.0)

N = FacetNormal(mesh)
Gext = p_right * inner(v, cofac(F)*N) * ds(2) #_bottom
R = inner(P,grad(v))*dx + Gext #dot(T,v)*ds(2)
#J = derivative(R,u,du)

set_log_level(PROGRESS)
solve(R == 0, u, bcs)


# Save solution in VTK format
file = File("displacement.pvd");
file << u;

#displacement of point (1,0.5,0.5)
point0 = np.array([10.0,0.5,0.5])
d0 = np.zeros(3)

u.eval(d0,point0)
print d0
