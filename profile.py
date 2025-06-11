import numpy as np
import matplotlib.pyplot as plt
import sys
class Joukowsky_transform:
	def __init__(self, c, a, t_0):
		print("Joukowsky_transform initialized.")
		self.c = c
		self.a = a
		self.t_0 = t_0
	def __call__(self, t):
		epsc = 1e-15+1j*1e-15
		z = 0.5*(t+self.t_0+self.c**2/(t+self.t_0+epsc)) 
		print("Joukowsky transform (t -> z) result z: ")
		print(z)
		return z
	def inverse(self, z):
		#!!!For the difficulties with computation of sqrt function see: p.20, p.33, Maklakov. Mentioned pages content explains the formula zmc = 1j*np.sqrt(self.c-z)*np.sqrt(self.c+z)
		#zmc = 1j*np.sqrt(self.c-z)*np.sqrt(self.c+z) !!! Correct upper part!
		#zmc = np.sqrt(z-self.c)*np.sqrt(z+self.c) !!! Another good one!
		zmc = 1j*np.sqrt(self.c-z)*np.sqrt(self.c+z)
		t = (z + zmc - self.t_0)
		t = np.where(np.real(t)**2+np.imag(t)**2>=self.a**2,  (z + zmc - self.t_0), (z - zmc - self.t_0)) # See Wikipedia about Joukowsky. Internal part of circle maps to lower half-plane, thats why it works.
		print("Inverse Joukowsky transform (z -> t) result t: ")
		print(t)
		return  t
class Joukowsky_airfoil_flow_potential:
	def __init__(self, u_inf, alpha, a, Gamma):
		print("Joukowsky_airfoil_flow_potential initialized.")
		self.u_inf = u_inf
		self.alpha = alpha
		self.a = a
		self.Gamma = Gamma
	def __call__(self, t):
		epsc = 1e-15+1j*1e-15
		w = 0.5*self.u_inf*(np.exp(-1*1j*self.alpha)*t+(np.exp(1j*self.alpha)*self.a**2)/t)+( self.Gamma/(2*np.pi*1j) )*np.log(t+epsc) #calculating flow potential 
		w = np.where( np.real(t)**2+np.imag(t)**2<self.a**2, 0.0+1j*0.0, w) #zeroing inner part of parametric circle
		print("Flow potential calculation result w: ")
		print(w)
		return w

def circle(X, Y, x_c, y_c, c, h, d): #make circle in parametric plane
	a = np.sqrt(c**2+h**2)+d
	Y_pos = np.sqrt(a**2-(X-x_c)**2)+y_c
	Y_neg = -np.sqrt(a**2-(X-x_c)**2)+y_c
	plt.scatter(x_c,y_c, c = 'g')
	return [Y_neg, Y_pos]
def Joukowsky(X, Y, c, attack):
	Z = X + 1j*Y #Create parametric plane
	JT = Joukowsky_transform(c, 0.0, 0.0) #Apply conformal map
	Z_J = JT(Z)
	Z_J = Z_J*np.exp(1j*attack)
	Re_Z_J = np.real(Z_J)
	Im_Z_J = np.imag(Z_J)
	return [Re_Z_J, Im_Z_J]

def airfoil(c, h, d, attack):
	print("Plotting airfoil:")
	a = np.sqrt(c**2+h**2)+d
	sigma = np.arctan(h/c)
	x_c = c - a*np.cos(sigma)
	y_c = a*np.sin(sigma)
	X = np.linspace(x_c-a, x_c+a, 1000) #parametric plane mesh
	Y = np.linspace(y_c-a, y_c+a, 1000)
	
	Y_circle = circle(X, Y, x_c, y_c, c, h, d)
	Re_Z_J_pos = Joukowsky(X, Y_circle[1], c, attack)[0] #apply conformal map to upper part of circle
	Im_Z_J_pos = Joukowsky(X, Y_circle[1], c, attack)[1]
	
	Re_Z_J_neg = Joukowsky(X, Y_circle[0], c, attack)[0] #apply conformal map to lower part of circle
	Im_Z_J_neg = Joukowsky(X, Y_circle[0], c, attack)[1]
	
	#print("y_pos")
	#print(Y_circle[1])
	#print("y_neg")
	#print(Y_circle[0])

	plt.axis('equal')
	plt.plot(X, Y_circle[0], "r--", alpha = 0.5)
	plt.plot(X, Y_circle[1], "b--", alpha = 0.5)
	plt.plot(Re_Z_J_neg, Im_Z_J_neg, "r-")
	plt.plot(Re_Z_J_pos, Im_Z_J_pos, "b-")

def flow(c, h, d, u_inf, alpha):
	print("Calculating flow:")
	a = np.sqrt(c**2+h**2)+d
	sigma = np.arctan(h/c)
	Gamma = -2*np.pi*a*u_inf*np.sin(alpha+sigma)
	print("Circulation Г = ", Gamma)
	x_c, y_c = (c - a*np.cos(sigma)), a*np.sin(sigma) #x_c, y_c -- coordinates of t' coord system zero in t
	
	#---Initializing parametrical plane t'--- 
	k = 2.0 #modifier for size of domain
	T1_x = np.linspace(x_c-k*a, x_c+k*a, 1000) #create parametrical plane t'
	T1_y = np.linspace(y_c-k*a, y_c+k*a, 1000)  
	T1_xmesh, T1_ymesh = np.meshgrid(T1_x, T1_y)
	T1_mesh = T1_xmesh + 1j*T1_ymesh
	t_0 = 1j*h+d*np.exp(1j*(np.pi-sigma))
	w = Joukowsky_airfoil_flow_potential(u_inf, alpha, a, Gamma) #initialize flow potential function in parametric plane t'
	J = Joukowsky_transform(c, a, t_0) # initialize Joukowsky transform t --> z
	
	#---Calculating flow potential in parametrical plane---
	W_mesh = w(T1_mesh) #Obtain flow potential in parametric plane
	Psi_param = np.imag(W_mesh)
	#plt.contour(T1_xmesh+np.real(t_0), T1_ymesh+np.imag(t_0), Psi_param, levels=100, cmap='viridis') #Solution in parametric plane, shifted to t plane.
	#plt.pcolormesh(T1_xmesh, T1_ymesh, Psi_param, shading='auto', cmap='viridis', vmin=-5, vmax=5)
	
	#---Calculating critical points in parametrical plane---
	z_1_param = ( -1*Gamma/(2*np.pi*1j) + np.sqrt( -1*Gamma**2/(4*np.pi**2) + u_inf**2*a**2 +1j*0.0))*np.exp(1j*alpha)*(1/u_inf)
	z_2_param = ( -1*Gamma/(2*np.pi*1j) - np.sqrt( -1*Gamma**2/(4*np.pi**2) + u_inf**2*a**2 +1j*0.0))*np.exp(1j*alpha)*(1/u_inf)
	
	#---Initilizing physical plane z---
	Z_mesh = T1_mesh
	#Z_mesh = T1_mesh+t_0 #z plane has same origin as t ??? why adding t_0 is not needed???
	Z_xmesh = np.real(Z_mesh)
	Z_ymesh = np.imag(Z_mesh)
	
	#---Calculating flow potential in physical plane---
	W_phys_mesh = w(J.inverse(Z_mesh)) #Obtain flow potential in physical plane.
	Psi_phys = np.imag(W_phys_mesh)
	plt.contour(Z_xmesh, Z_ymesh, Psi_phys, levels=500, colors='black', linestyles='solid', linewidths=0.5) #Solution in parametric plane, shifted to t plane.
	#plt.pcolormesh(Z_xmesh, Z_ymesh, Psi_phys, shading='auto', cmap='viridis')
	#plt.colorbar()
	
	#---Calculating velocity field (p. 8, p. 17 Maklakov)---
	dz = 1e-10+1j*1e-10
	dW_phys_mesh = w(J.inverse(Z_mesh+dz))-W_phys_mesh #f(z+h)-f(z)
	dWdz_phys_mesh = dW_phys_mesh/dz #(f(z+h)-f(z))/h
	u_x = np.real(dWdz_phys_mesh)
	u_y = -1*np.imag(dWdz_phys_mesh)
	u_mag = np.sqrt(u_x**2+u_y**2)
	plt.pcolormesh(Z_xmesh, Z_ymesh, u_mag, shading='auto', cmap='viridis')
	plt.colorbar(label = "Скорость")
	stride = 25
	plt.quiver(Z_xmesh[::stride, ::stride], Z_ymesh[::stride, ::stride], u_x[::stride, ::stride], u_y[::stride, ::stride], width = 0.001)
	
	#---Calculating critical points in physical plane---
	if (Gamma > 4*np.pi*u_inf):
		print("Case Gamma > 4*np.pi*u_inf")
		z_1 = J(z_1_param) #Mapping to physical plane after evaluating
		z_2 = J(z_2_param) #Mapping to physical plane after evaluating
	if (Gamma == 4*np.pi*u_inf):
		print("Case Gamma == 4*np.pi*u_inf")
		z_1 = J(z_1_param)
		z_2 = z_1 #z_2 and z_1 are multiple roots
	if (Gamma < 4*np.pi*u_inf):
		print("Case Gamma < 4*np.pi*u_inf")
		z_1 = J(z_1_param) #Mapping to physical plane after evaluating
		z_2 = J(z_2_param) #Mapping to physical plane after evaluating
	
	plt.scatter(np.real(z_1), np.imag(z_1), color = "r")
	plt.scatter(np.real(z_2), np.imag(z_2), color = "b")


	return 0

def main(args):
	da = {"deg":args[0], "c": args[1], "h": args[2], "d": args[3]}
	if (process_argv(sys.argv, da)):
		return
	print("Arguments:\n", da)
	deg = da["deg"]
	c = da["c"]
	h = da["h"]
	d = da["d"]

	
	angle = deg*np.pi/180
	print("Angle = ", angle, " rad.")
	plt.figure()
	plt.grid()
	flow(c, h, d, 1.0, angle)
	airfoil(c, h, d, 0.0)

def process_argv(argv, da):
	argc = len(argv)
	if (argc == 1):
		print("You have not used any flags. Use 'default' flag to load values from program;\n"\
		"Write in format:\n python3 *program*.py deg *degrees* c *c* h *h* d *d*\n" \
		"or to change only attack angle use python3 *program*.py *deg* ")
		return 1
	if (argc>1 and argv[1] == "default"):
		return 0
	if (argc==2 and argv[1] != "default"):
		da["deg"] = float(argv[1])
		return 0
	if (argc>2):
		for i_arg in range(argc):
			if (argv[i_arg] == "deg"):
				da["deg"] = float(argv[i_arg+1])
			if (argv[i_arg] == "c"):
				da["c"] = float(argv[i_arg+1])
			if (argv[i_arg] == "h"):
				da["h"] = float(argv[i_arg+1])
			if (argv[i_arg] == "d"):
				da["d"] = float(argv[i_arg+1])
		return 0
	return 1

if __name__=="__main__":
	args = [0.0, 1.2, 0.2, 0.1] #default args
	main(args)
	plt.show()
