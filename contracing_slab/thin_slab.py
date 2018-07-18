import matplotlib.pyplot as plt
import numpy as np
from guccionematerial import *
from fenics import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4


mesh = UnitCubeMesh(4,4,4)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

boundary_markers = MeshFunction("size_t", mesh,mesh.topology().dim() - 1)
boundary_markers.set_all(0)
left.mark(boundary_markers, 1)
right.mark(boundary_markers, 2)

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


"""
#fiber orientation - not needed for isotropic models
f0 = as_vector([ 1.0, 0.0, 0.0 ])
s0 = as_vector([ 0.0, 1.0, 0.0 ])
n0 = as_vector([ 0.0, 0.0, 1.0 ])
"""

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # the right Cauchy-Green tensor
E = 0.5*(C - I)             # the Green-Lagrange strain tensor

#define the strain energy
#material = GuccioneMaterial(e1=f0, e2=s0, e3=n0,
#                               C=2.0, bf=8, bt=2, bfs=4,kappa=1.0e4,Tactive=0.0)
#psi = material.strain_energy(F)

mu    = 4.0  # N  / m^2 Lame parameter mu for PVC plastic
lmbda = 20.0  # N  / m^2 Lame parameter lambda for PVC plastic

E = variable(E)
psi = lmbda/2*(tr(E)**2) + mu*tr(E*E)


#first Piola-Kirchoff stress
S = diff(psi,E)
P = F*S

#P = diff(psi,F)

p_right = Constant(0.0)

N = FacetNormal(mesh)
Gext = p_right * inner(v, cofac(F)*N) * ds(2) #_bottom
R = inner(P,grad(v))*dx + Gext #dot(T,v)*ds(2)
#J = derivative(R,u,du)

#set_log_level(PROGRESS)

#file = File("displacement.pvd");


#solve(R == 0, u,bcs)

#displacement of point (1,0.5,0.5)

#d0 = np.zeros(3)
#disp = np.zeros((5,3))
#print disp[:,0]

i = 0

load_steps = 6
target_load = 10.0

point0 = np.array([1.0,0.5,0.5])
disp = np.zeros((load_steps,3))
d0 = np.zeros(3)

loads = np.linspace(0,target_load,load_steps)
print loads 

for step in range(load_steps):
    load = -loads[step]
    print load
    p_right.assign(load)
    #Gext = p_right * inner(v, cofac(F)*N) * ds(2) #_bottom
    #R = inner(P,grad(v))*dx + Gext #dot(T,v)*ds(2)
    #print p_right
    solve(R == 0, u, bcs)
    #u.eval(d0,point0)
    u.eval(disp[step,:],point0)
    disp[i,:] = d0
    #print(d0)
    #print type(d0), type(d0[0])
    #disp.append(d0.copy())
    # Save solution in VTK format
    #file << u;
    
    
#disp = np.asarray(disp)


plt.plot(disp[:,0],loads)



plt.show()

#print disp
