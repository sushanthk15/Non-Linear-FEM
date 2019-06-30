def material_routine(epsilon, epsilon_plastic_old, lamda, mu , sigmaY, youngs_modulus):
	C_ijkl = (2*mu+lamda)*np.eye(3) #np.eye(3) = 3x3 Identity Matrix
	sigma_trial = np.dot(C_ijkl,(epsilon - epsilon_plastic_old)) #3x1 Matrix
	sigma_trial_dev = sigma_trial - np.sum(sigma_trial)/3
	sigma_trial_eq = np.sqrt((1.5*np.dot((np.transpose(sigma_trial_dev)),sigma_trial_dev))) #Scalar
	
	
	
	#Yield Condition ------> Check for Elastic and Plastic Case 
	if (sigma_trial_eq - sigmaY) < 0:
		print("This is in Elastic Region")
		#C_t = np.array([(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda)]).reshape(3,3)
		plastic_multiplier = 0
		#sigma_3d = sigma_trial
	else :
		print("This is in  Plastic Region")
		plastic_multiplier = sigma_trial_eq /(3*mu + sigmaY)
	
	# Stress, C_t and epsilon_plastic_old update
	sigma_new = np.sum(sigma_trial)/3 + ((sigma_trial_eq - 3*mu*plastic_multiplier)/sigma_trial_eq)*sigma_trial_dev
	
	C_t = ((lamda - 3*mu)/3)*np.eye(3) + (2*mu*(sigma_trial_eq -3*mu*plastic_multiplier)/sigma_trial_eq)*(2/3*np.eye(3)) - 3*mu*np.dot(sigma_trial_dev, np.transpose(sigma_trial_dev))
	
	epsilon_plastic_new = epsilon_plastic_old + plastic_multiplier*np.sign(sigma_new)
	
	return sigma_new, C_t, epsilon_plastic_new
