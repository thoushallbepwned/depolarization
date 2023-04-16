import numpy as np


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

    min_bound = np.min(opinion_distribution)
    max_bound = np.max(opinion_distribution)
    threshold = (max_bound + min_bound) / 2


    pop_distribution = np.zeros((2, d))
    pop_distribution[0] = np.sum(opinion_distribution < threshold, axis=0) / n
    pop_distribution[1] = np.sum(opinion_distribution > threshold, axis=0) / n

    # Calculate Delta A in each dimension
    delta_a = np.abs(pop_distribution[1] - pop_distribution[0])

    # Calculate the center of gravity in each dimension
    center_of_gravity = np.zeros((2, d))
    center_of_gravity[0] = np.sum(
        opinion_distribution[opinion_distribution < threshold] * np.abs(opinion_distribution[opinion_distribution < threshold]),
        axis=0) / np.sum(np.abs(opinion_distribution[opinion_distribution < threshold]), axis=0)
    center_of_gravity[1] = np.sum(
        opinion_distribution[opinion_distribution > threshold] * np.abs(opinion_distribution[opinion_distribution > threshold]),
        axis=0) / np.sum(np.abs(opinion_distribution[opinion_distribution > threshold]), axis=0)

    # Calculate the polarization metric in each dimension
    polarization = np.abs(center_of_gravity[1] - center_of_gravity[0]) / delta_a

    # Printing results
    #print("polarization is:", polarization)
    #print( "Threshold:",threshold,"\n",
    #       "max bound:", max_bound, "\n",
   #        "min_bound:", min_bound, "\n")

    return polarization