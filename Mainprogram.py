#Import useful libraries
import numpy as np
import matplotlib.pyplot as plt

#Defining Parameters:

youngs_modulus = 200e9 #PA
poisson_ratio = 0.30
sigmaY = 400e6 #Pa
r_in = 20e-6 #m
r_out = 100e-6 #m
epsilon_v = 0.02
final_time = 1 #seconds
delta_t = 0.1 #seconds
time_steps = 10
number_of_elements = 10
number_of_nodes = number_of_elements + 1

#Derived parameters:

dr = (r_out - r_in)/number_of_elements
lamda = (poisson_ratio * youngs_modulus)/((1 - 2*poisson_ratio)*(1+poisson_ratio))
mu  = youngs_modulus/ (2*(1+poisson_ratio))
epsilon_v_init = (1+poisson_ratio)*(sigmaY/youngs_modulus)

#Initializing all the measurable quantities:

u = np.zeros((number_of_nodes,1)) #Displacements at every node
u_real = np.zeros((number_of_nodes,1))
epsilon_plastic = np.zeros((3,1))
epsilon_3D = np.zeros((3,1)) # Strain in Voigt Notation for the given problem is 3x1 Matrix
stress_3D = np.zeros((3,1)) # Stress in Voigt Noation for the given problem is 3x1 Matrix
Material_Stiffness = np.zeros((3,3)) #Stress = Stiffness x Strain --> C = 3x3 Matrix
Kt = np.zeros((number_of_nodes,number_of_nodes))
F_int = np.zeros((number_of_nodes,1))
F_ext = np.zeros((number_of_nodes,1))
delta_u = np.zeros((number_of_nodes,1))

#Storing the measured quantities for tracking its history

u_history = [u]
Kt_history = [Kt]
F_int_history = [F_int]
F_ext_history = [F_ext]
epsilon_plastic_history = [epsilon_plastic]
stress_history = []

#Defining the elements:

r = np.zeros((number_of_elements,2))
tr = r_in
for i in range(number_of_elements):
    r[i,0] = tr
    r[i,1] = tr+dr
    tr = r[i,1]

# Material Routine:

def material_routine(epsilon, epsilon_plastic_old, lamda, mu , sigmaY, youngs_modulus):
    C_ijkl = (2*mu+lamda)*np.eye(3) #np.eye(3) = 3x3 Identity Matrix
    sigma_trial = np.dot(C_ijkl,(epsilon - epsilon_plastic_old)) #3x1 Matrix
    sigma_trial_dev = sigma_trial - np.sum(sigma_trial)/3
    sigma_trial_eq = np.linalg.norm((1.5*np.dot((np.transpose(sigma_trial_dev)),sigma_trial_dev))) #Scalar
    
    if sigma_trial_eq ==0:
		C_t1 = np.array([(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda)]).reshape(3,3)
		sigma_new = 2*mu*(epsilon - epsilon_plastic_old) + lamda*np.sum(epsilon - epsilon_plastic_old)
		epsilon_plastic_new = epsilon_plastic_old
	else:
		
		#Yield Condition ------> Check for Elastic and Plastic Case 
		if (sigma_trial_eq - sigmaY) < 0:
			print("This is in Elastic Region")
			#C_t1 = np.array([(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda)]).reshape(3,3)
			plastic_multiplier = 0
			#sigma_3d = sigma_trial
		else :
			print("This is in  Plastic Region")
			plastic_multiplier = sigma_trial_eq /(3*mu + sigmaY)
		
		# Stress, C_t and epsilon_plastic_old update
		
		sigma_new = np.sum(sigma_trial)/3 + ((sigma_trial_eq - 3*mu*plastic_multiplier)/sigma_trial_eq)*sigma_trial_dev
		
		C_t = ((3*lamda + 2*mu)/3)*np.eye(3) + (2*mu*(sigma_trial_eq -3*mu*plastic_multiplier)/sigma_trial_eq)*(2/3*np.eye(3)) - 3*mu*np.dot(sigma_trial_dev, np.transpose(sigma_trial_dev))/(sigma_trial_eq)**2
		#C_t1 = np.array([(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda)]).reshape(3,3)
		epsilon_plastic_new = epsilon_plastic_old + plastic_multiplier*np.sign(sigma_new)
    
    return sigma_new, C_t, epsilon_plastic_new #,C_t1, C_ijkl
	
#Defining my Element Routine:

def elementroutine(r,lamda, mu,u_element, epsilon_plastic,sigmaY,youngs_modulus):
    length_of_element = r[1]-r[0]
    jacobian = length_of_element/2
    weight = 2
    gauss_point = 0
    N1 = (1-gauss_point)/2
    N2 = (1+gauss_point)/2
    B = np.array([-1/(2*jacobian), 1/(2*jacobian), N1/(N1*r[0] + N2*r[1]), N2/(N1*r[0] + N2*r[1]), N1/(N1*r[0] + N2*r[1]), N2/(N1*r[0] + N2*r[1])]).reshape(3,2)
	epsilon = np.dot(B, u_element)
	stress_new , Material_Stiffness1, strain_plastic = material_routine(epsilon, epsilon_plastic, lamda, mu,sigmaY,youngs_modulus)
    #C = np.array([(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda)]).reshape(3,3)
    Kt_element = weight * ((B.transpose()).dot(Material_Stiffness1).dot(B)) * (N1*r[0] + N2*r[1])**2 * jacobian
	F_int = weight* ((B.transpose()).dot(stress_new))* (N1*r[0] + N2*r[1])**2 * jacobian
	F_ext = (N1*r[0] + N2*r[1])*stress_new[0]*(np.array([ N1,N2]).reshape(2,1))
    return Kt_element, F_int, F_ext, stress_new, strain_plastic


#Assembling Matrix
def assembling(number_of_nodes,num):
    A=np.zeros((2,number_of_nodes))
    A[0][num]=1
    A[1][num+1]=1
    return A	
	
#Declaring my time interval:

all_times = np.linspace(0,final_time+delta_t, time_steps)

#Plasticity initiation:
#esilonV_init = already declared above

#Time Integration

for step in all_times:
	u_time = u_history[-1]
	epsilon_plastic_history_time = [epsilon_plastic_history[-1]]
	stress_history_time = [stress_history[-1]]
	
	for k in range(5): #Newton-Raphson Method converges within 5 iterations
		if k>0:
			if ((np.amax(np.absolute(reduced_G))) <= 0.005*(np.amax(np.absolute(F_int))) and np.amax(np.absolute(reduced_delta_u)) <= 0.005*(np.amax(np.absolute(u_time[1:,0])))):
				reduced_Kt =0*reduced_Kt
				reduced_G =0*reduced_G
			u_time[0,0] = step*epsilon_v*r_in/3
			for j in range(number_of_elements):
				A =  assembling(number_of_nodes,j)
				u_element = np.dot(A,u_time)
				#print(epsilon_plastic[-1].shape)
				Kt_ele, F_int_ele,F_ext_ele, stress3D2, strain_plastic1 = elementroutine(r[j],lamda, mu,u_element,epsilon_plastic_history_time[-1],sigmaY,youngs_modulus)
				#print(stress3D2.shape)
				epsilon_plastic_history_time.append(strain_plastic1)
				stress_history_time.append(stress3D2)
				#Assembling
				Kt=Kt + ((A.transpose()).dot(Kt_ele).dot(A))
				F_int=F_int+((A.transpose()).dot(F_int_ele))
				F_ext=F_ext+((A.transpose()).dot(F_ext_ele)) 
			#reduced_u = u_time[1:,0]
			reduced_Kt = Kt[1:,1:]
			reduced_G = F_int[1:,0] - F_ext[1:,0]
			reduced_delta_u = np.linalg.solve(reduced_Kt,reduced_G)
			u_time[1:,0] += reduced_delta_u
		stress_history.append(stress_history_time[-1])
		epsilon_plastic_history.append(epsilon_plastic_history_time[-1])
		u_history.append(u_time)
	f = open("NLFEMASSIGNMENT.txt", "a")
	print("Kt for time step = ",step,"\n", Kt, file=f)
	print("Fint for time step = ",step,"\n", F_int, file=f)
	#print("The Euler Angles [phi_1, phi, phi_2]  as follows \n",BungeAngles, file=f)
	f.close()
	
			
	
	
