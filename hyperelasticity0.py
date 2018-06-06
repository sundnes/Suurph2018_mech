import matplotlib.pyplot as plt
import numpy as np
from fenics import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4
#parameters["allow_extrapolation"] = True


L = 10
W = 1

mesh = BoxMesh(Point(0,0,0), Point(L,W,W),20,2,2)

V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0)
bottom = CompiledSubDomain("near(x[2], side) && on_boundary", side = 0)

boundary_markers = MeshFunction("size_t", mesh,mesh.topology().dim() - 1)
boundary_markers.set_all(0)
left.mark(boundary_markers, 1)
bottom.mark(boundary_markers, 2)

# Redefine boundary measure
ds = Measure('ds',domain=mesh,subdomain_data=boundary_markers)

# Define Dirichlet boundary (x = 0 or x = 1)
clamp = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, clamp, left)
bcs = [bc]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
T  = Constant((0.0,  0.0, 0.001))  # Traction force on the boundary

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient

F = variable(F)
C = F.T*F                   # Right Cauchy-Green tensor
# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
mu, lmbda = Constant(6.0), Constant(4.0)
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

#first Piola-Kirchoff stress
P = diff(psi,F)

p_bottom = 0.004
N = FacetNormal(mesh)
Gext = p_bottom * inner(v, cofac(F)*N) * ds(2) 
R = inner(P,grad(v))*dx + Gext 

solve(R == 0, u, bcs)

set_log_level(PROGRESS)

#displacement of point (1,0.5,0.5):
point0 = np.array([10.0,0.5,0.5])
d0 = np.zeros(3)
u.eval(d0,point0)
print d0

# Save solution in VTK format
file = File("displacement.pvd");
file << u;



