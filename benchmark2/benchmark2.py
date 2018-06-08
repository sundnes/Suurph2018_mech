import matplotlib.pyplot as plt

from geometry_utils import load_geometry_from_h5

#from geometry import *
#from dolfin import *

from dolfin import *

# Adjust log level
set_log_level(PROGRESS)
parameters["form_compiler"]["cpp_optimize"] = True

geo = load_geometry_from_h5('simple_ellipsoid.h5')

mesh = geo.mesh

V = VectorFunctionSpace(geo.mesh,'P',1)

new_ffun = MeshFunction("size_t", mesh, 2, mesh.domains())
print set(new_ffun.array())

exit()

"""
base =  CompiledSubDomain("near(x[0], -5) && on_boundary")
endo = CompiledSubDomain("pow(x[1]/rs,2.0)+pow(x[2]/rs,2.0)+pow(x[0]/rl,2.0) < tol && on_boundary",rs=7,rl=17,tol=1e-3)

boundary_markers = MeshFunction("size_t", mesh,mesh.topology().dim() - 1)
boundary_markers.set_all(0)
base.mark(boundary_markers, 1)
endo.mark(boundary_markers, 2)
"""


# Redefine boundary measure
ds = Measure('ds',domain=mesh,subdomain_data=boundary_markers)

clamp = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, clamp, base)
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


#endo = geo.markers['ENDOPT'][0]
#print endo
p_endo = 1.0e2
# Define nonlinear problem


N = FacetNormal(geo.mesh)
Gext = p_endo * inner(v, cofac(F)*N) * ds(2) #_bottom
R = inner(P,grad(v))*dx + Gext #dot(T,v)*ds(2)
#J = derivative(R,u,du)

solve(R==0,u,bcs)

# Save solution to file
file = File("displacement.pvd")
file << u


#print ds.subdomain_data()
#fixed_base = DirichletBC(V,Constant(0,0,0),
#plt.show()

