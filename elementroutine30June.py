def elementroutine(r,lamda, mu,u_element, epsilon_plastic,sigmaY,youngs_modulus):
    length_of_element = r[1]-r[0]
    jacobian = length_of_element/2
    weight = 2
    gauss_point = 0
    N1 = (1-gauss_point)/2
    N2 = (1+gauss_point)/2
    B = np.array([-1/(2*jacobian), 1/(2*jacobian), N1/(N1*r[0] + N2*r[1]), N2/(N1*r[0] + N2*r[1]), N1/(N1*r[0] + N2*r[1]), N2/(N1*r[0] + N2*r[1])]).reshape(3,2)
	epsilon = np.dot(B, u_element)
	stress_new , Material_Stiffness1, strain_plastic = materialroutine(epsilon, epsilon_plastic, lamda, mu,sigmaY,youngs_modulus)
    #C = np.array([(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda), lamda, lamda,lamda,(2*mu + lamda)]).reshape(3,3)
    Kt_element = weight * ((B.transpose()).dot(Material_Stiffness1).dot(B)) * (N1*r[0] + N2*r[1])**2 * jacobian
	F_int = weight* ((B.transpose()).dot(stress_new))* (N1*r[0] + N2*r[1])**2 * jacobian
	F_ext = (N1*r[0] + N2*r[1])*stress_new[0]*(np.array([ N1,N2]).reshape(2,1))
    return Kt_element, F_int, F_ext, stress_new, strain_plastic