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
	Z_J = Z_J*np.exp(-1j*attack)
	Re_Z_J = np.real(Z_J)
	Im_Z_J = np.imag(Z_J)
	return [Re_Z_J, Im_Z_J]

def airfoil(c, h, d, attack, meshsize):
	print("Plotting airfoil:")
	a = np.sqrt(c**2+h**2)+d
	sigma = np.arctan(h/c)
	x_c = c - a*np.cos(sigma)
	y_c = a*np.sin(sigma)
	X = np.linspace(x_c-a, x_c+a, meshsize) #parametric plane mesh
	Y = np.linspace(y_c-a, y_c+a, meshsize)
	
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
	#plt.plot(X, Y_circle[0], "r--", alpha = 0.5)
	#plt.plot(X, Y_circle[1], "b--", alpha = 0.5)
	#plt.plot(Re_Z_J_neg, Im_Z_J_neg, "r")
	#plt.plot(Re_Z_J_pos, Im_Z_J_pos, "b")
	plt.scatter(Re_Z_J_neg, Im_Z_J_neg, c="r")
	plt.scatter(Re_Z_J_pos, Im_Z_J_pos, c="b")
	return [Re_Z_J_neg, Im_Z_J_neg, Re_Z_J_pos, Im_Z_J_pos]
def outer(c, h, d, k): # k - multiplier for size of domain
	a = np.sqrt(c**2+h**2)+d
	sigma = np.arctan(h/c)
	x_c = c - a*np.cos(sigma)
	y_c = a*np.sin(sigma)
	x, y = [0.0]*4, [0.0]*4
	x[0], y[0] = k*c, 0.0
	x[1], y[1] = 0.0, k*c
	x[2], y[2] = -k*c,0.0
	x[3], y[3] = 0.0, -k*c
	plt.scatter(x, y)
	return np.array(x), np.array(y)

def step1(filename, profile_points, div):
	Re_Z_J_neg, Im_Z_J_neg, Re_Z_J_pos, Im_Z_J_pos = profile_points[0], profile_points[1], profile_points[2], profile_points[3]
	with open(filename, 'a+') as f:
		f.write(f"//Lower part of profile\n")
		for i in range(div-1):
			f.write(f"Point({i+1}) = {{{Re_Z_J_neg[i]}, {Im_Z_J_neg[i]}, 0, 1.0}};\n") #Writing to geo file
		f.write(f"//Upper part of profile\n")
		for i in range(div-1):
			f.write(f"Point({div+i}) = {{{Re_Z_J_pos[-(i+1)]}, {Im_Z_J_pos[-(i+1)]}, 0, 1.0}};\n") #Writing to geo file

def step2(filename, outer_points, div):
	x = outer_points[0]
	y = outer_points[1]
	with open(filename, 'a+') as f:
		f.write(f"//Circle\n")
		for i in range(4):
			f.write(f"Point({2*div-1+i}) = {{{x[i]}, {y[i]}, 0, 1.0}};\n") #Writing to geo file
		f.write(f"Point({2*div+3}) = {{0.0, 0.0, 0, 1.0}};\n")
def step3(filename, div):
	with open(filename, 'a+') as f:
		f.write(f"//Profile lines\n")
		for i in range(1,2*div-2):
			f.write(f"Line({i}) = {{{i}, {i+1}}};\n") #Writing to geo file
		f.write(f"Line({2*div-2}) = {{{2*div-2}, {1}}};\n")
def step4(filename, div):
	with open(filename, 'a+') as f:
		f.write(f"//Creating profile curve loop\n")
		f.write(f"lines_ind[] = {{1 : {2*div-2}}};\n")
		f.write(f"Curve Loop(1) = lines_ind[];\n")
		f.write(f"Circle({2*div-1}) = {{{2*div}, {2*div+3}, {2*div+2}}};\n")
		f.write(f"Circle({2*div}) = {{{2*div+2}, {2*div+3}, {2*div}}};\n")
		f.write(f"Curve Loop(2) = {{{2*div-1}}};\n")
		f.write(f"Curve Loop(3) = {{{2*div}}};\n")
		f.write(f"//Creating plane\n")
		f.write(f"Plane Surface(1) = {{1, 2, 3}};\n")
		f.write(f"//Physical groups\n")
		f.write(f"Physical Curve(\"Left\") = {{{2*div-1}}};\n")
		f.write(f"Physical Curve(\"Right\") = {{{2*div}}};\n")
		f.write(f"Physical Curve(\"Profile\") = lines_ind[];\n")
		f.write(f"Physical Surface(\"Domain\") = {{1}};\n")
		f.write(f"Mesh.Algorithm = 2;\n")
		f.write(f"Mesh.MshFileVersion = 2.2;\n")

#1) c = 1.0 h = 0.0 d = 0.1
#2) c = 1.0 h = 0.1 d = 0.05

def main(args, div, tag): 
	da = {"deg":args[0], "c": args[1], "h": args[2], "d": args[3]}
	deg = da["deg"]
	c = da["c"]
	h = da["h"]
	d = da["d"]

	angle = deg*np.pi/180
	print("Angle = ", angle, " rad.")
	plt.figure()
	plt.grid()
	profile_points = airfoil(c, h, d, angle, div)
	outer_points = outer(c, h, d, 8)
	fname = f"jouk_geo_profile{tag}_div{div}_angle{deg}.geo"
	step1(fname, profile_points, div)
	step2(fname, outer_points, div)
	step3(fname, div)
	step4(fname, div)

profile1 = [0.0, 1.0, 0.0, 0.1]
profile2 = [0.0, 1.0, 0.1, 0.05]
profile3 = [15.0, 1.0, 0.1, 0.05]
main(profile1, 50, 1)
main(profile1, 100, 1)
main(profile1, 150, 1)
main(profile2, 150, 2)
main(profile3, 150, 3)
plt.show()


