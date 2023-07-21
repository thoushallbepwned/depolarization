import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def polarization_metric(df):
    """
    Calculates the polarization metric for a d-dimensional opinion vector distribution.

    Parameters:
        df (pandas dataframe): A pandas dataframe representing the opinion distribution
                                            of n individuals in a d-dimensional space.

    Returns:
        polarization (numpy array): A 1 x d numpy array representing the polarization metric in each
                                    dimension of the opinion space.
    """
    n, d = df.shape

    # Convert dataframe to numpy array
    opinion_distribution = df.to_numpy()

    # Calculate the population distribution in each dimension

    # Setting the boundary to not be centered at 0
    #
    # min_bound = np.min(opinion_distribution)
    # max_bound = np.max(opinion_distribution)
    # threshold = (max_bound + min_bound) / 2
    threshold = 0

    pop_distribution = np.zeros((2, d))
    pop_distribution[0] = np.sum(opinion_distribution < threshold, axis=0) #/ n
    pop_distribution[1] = np.sum(opinion_distribution > threshold, axis=0) #/ n

    # print("pop_distribution is:", pop_distribution)
    #print("pop_distribution[0] is:", pop_distribution[0])
    # print("pop_distribution[1] is:", pop_distribution[1])

    # Calculate Delta A in each dimension
    delta_a = np.abs(pop_distribution[1] - pop_distribution[0])/n

    # Calculate the center of gravity in each dimension
    #print("This is the opinion distribution", opinion_distribution.shape)
    cog_minus = np.zeros((d,1))
    cog_plus = np.zeros((d,1))
    for dim in range(d):
        dim = dim - 1
        for i in opinion_distribution[:,dim]:
            if i < threshold:
                cog_minus[dim] += i
            else:
                cog_plus[dim] += i

    #print("What is the added values of cog_minus", cog_minus)
    #print(cog_plus)

    cog_minus = cog_minus.T / pop_distribution[0]
    cog_plus = cog_plus.T/pop_distribution[1]
    #("what is the new cog_minus?", cog_minus)

    distance = np.abs(cog_minus - cog_plus)/2
    #print("this is the distance", distance)
    # Calculate the polarization metric in each dimension
    polarization = (1-delta_a)*distance

    polarization[np.isnan(polarization)] = 0

    #adding a penalty term for variance

    variance  = np.var(opinion_distribution, axis = 0)

    #accounting for proportionality

    variance_penalty = variance# / np.max(variance)

    #polarization = polarization + variance_penalty

    # Fit a GaussianMixture model to the data

    bic_penalty = np.zeros(d)
    for dim in range(d):
        # Reshape the data to fit the GMM
        dim_data = opinion_distribution[:, dim].reshape(-1, 1)

        # Fit a GaussianMixture model to the data of the current dimension
        gmm = GaussianMixture(n_components=1)
        gmm.fit(dim_data)

        # Use the BIC to find the optimal number of components
        bic = gmm.bic(dim_data)

        # Calculate a penalty term based on the BIC and add it to the array
        bic_penalty[dim] = bic

    #bic_penalty = bic_penalty / np.max(bic_penalty)
    #polarization = polarization + bic_penalty


    #print("polarization aspect", polarization)
    #print("variance penalty", variance_penalty)
    #print("BIC aspect", bic_penalty)

    polarization = (polarization + variance_penalty)/2

    #  #np.abs(center_of_gravity[1] - center_of_gravity[0]) / delta_a
   #  print("distance is:", distance)
   #  print("delta_a is:", delta_a)
   # #Printing results
   #  print("polarization is:", polarization)
    #print( "Threshold:",threshold,"\n",
    #      "max bound:", max_bound, "\n",
    #      "min_bound:", min_bound, "\n")

    return polarization[0]