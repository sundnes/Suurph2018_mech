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
bottom = CompiledSubDomain("near(x[0], side) && on_boundary", side = L)

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

J = det(F)
C = pow(J, -float(2)/3) * F.T*F
E = 0.5*(C - I)

b_f = Constant(8)
b_t = Constant(2)
b_fs = Constant(4)
Q = b_f*E[0,0]**2 + b_t*(E[1,1]**2+E[2,2]**2+E[1,2]**2+E[2,1]**2) \
                  + b_fs*(E[0,1]**2+E[1,0]**2+E[0,2]**2+E[2,0]**2)

CC = Constant(2.0)
kappa = Constant(1.0e3)
psi = (CC/2)*(exp(Q -1)) + kappa * (J*ln(J) -J +1) 

#first Piola-Kirchoff stress
P = diff(psi,F)

p_bottom = Constant(0.004) 
N = FacetNormal(mesh)
Gext = p_bottom * inner(v, cofac(F)*N) * ds(2) 
R = inner(P,grad(v))*dx + Gext 

solve(R == 0, u, bcs)

#set_log_level(PROGRESS)

#displacement of point (10,0.5,0.5):
point0 = np.array([10.0,0.5,0.5])
d0 = np.zeros(3)
u.eval(d0,point0)
print d0

# Save solution in VTK format
file = File("displacement.pvd");
file << u;



