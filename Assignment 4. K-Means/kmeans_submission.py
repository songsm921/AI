import os
import math
from utils import converged, plot_2d, plot_centroids, read_data, \
    load_centroids, write_centroids_tofile
import matplotlib.pyplot as plt
import numpy as np


# problem for students
def euclidean_distance(dp1, dp2):
    """Calculate the Euclidean distance between two data points.

    Arguments:
        dp1: a list of floats representing a data point
        dp2: a list of floats representing a data point

    Returns: the Euclidean distance between two data points
    """
    point_1 = np.array(dp1)
    point_2 = np.array(dp2)
    dist = np.linalg.norm(point_1-point_2)
    return dist
    #pass


# problem for students
def assign_data(data_point, centroids):
    """Assign a single data point to the closest centroid. You should use
    the euclidean_distance function (that you previously implemented).

    Arguments:
        data_point: a list of floats representing a data point
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a string as the key name of the closest centroid to the data point
    """
    dist = []
    for k in centroids:
        data_point2 = centroids[k]
        dist.append(euclidean_distance(data_point,data_point2))
    for k in centroids:
        test = centroids[k]
        if min(dist) == euclidean_distance(data_point,test):
            return k
    
     



# problem for students
def update_assignment(data, centroids):
    """Assign all data points to the closest centroids. You should use
    the assign_data function (that you previously implemented).

    Arguments:
        data: a list of lists representing all data points
        centroids: a dictionary representing the centroids where the keys are
                   strings (centroid names) and the values are lists of
                   centroid locations

    Returns: a new dictionary whose keys are the centroids' key names and
             values are lists of points that belong to the centroid. If a
             given centroid does not have any data points closest to it,
             do not include the centroid in the returned dictionary.
    """
    ret = dict()
    for point in data:
        key = assign_data(point,centroids)
        if key not in ret:
            ret[key] = [point]
        elif key in ret:
            ret[key].append(point)
    #print(ret)
    return ret
    #pass
            

# problem for students
def mean_of_points(data):
    """Calculate the mean of a given group of data points. You should NOT
    hard-code the dimensionality of the data points).

    Arguments:
        data: a list of lists representing a group of data points

    Returns: a list of floats as the mean of the given data points
    """
    sum = [0 for i in range(len(data[0]))]
    n=0
    for point in data:
        sum = [point[i] + sum[i] for i in range(len(sum))]
        n=n+1   
    sum = np.array(sum)
    sum = sum/n
    sum = sum.tolist()
    sum = [float(i) for i in sum]
    return sum
    
        
    #pass


# problem for students
def update_centroids(assignment_dict):
    """Update centroid locations as the mean of all data points that belong
    to the cluster. You should use the mean_of_points function (that you
    previously implemented).

    Arguments:
        assignment_dict: the dictionary returned by update_assignment function
    
    Returns: A new dictionary representing the updated centroids
    """
    ret = dict()
    for item in assignment_dict:
        ret[item] = mean_of_points(assignment_dict[item])
    return ret  
    #pass

def main(data, init_centroids):
    #######################################################
    # You do not need to change anything in this function #
    #######################################################
    centroids = init_centroids
    old_centroids = None
    step = 0
    while not converged(centroids, old_centroids):
        # save old centroid
        old_centroids = centroids
        # new assignment
        assignment_dict = update_assignment(data, old_centroids)
        # update centroids
        centroids = update_centroids(assignment_dict)
        # plot centroid
        fig = plot_2d(assignment_dict, centroids)
        plt.title(f"step{step}")
        fig.savefig(os.path.join("results", "2D", f"step{step}.png"))
        plt.clf()
        step += 1
    print(f"K-means converged after {step} steps.")
    return centroids


if __name__ == '__main__':
    data, label = read_data("data/data_2d.csv")
    init_c = load_centroids("data/2d_init_centroids.csv")
    final_c = main(data, init_c)
    write_centroids_tofile("2d_final_centroids.csv", final_c)
