
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# ------- Helper Functions ---------
# Function to read the dataset from a file
def read_dataset(filename):
    return np.genfromtxt(filename, delimiter=',')

# Funtion to reduce dimensionality using svd
def reduce_dimension(data):
    if data.shape[1] <= 2:
        return data  # No need for PCA if dimensionality is 2 or less

    centered_data = data - np.mean(data, axis=0)
    u, sigma, vt = np.linalg.svd(centered_data)

    projected_data = centered_data @ vt[:2, :].T   # first two rows of vt

    return projected_data

# Function to plot data points along two principal axes
def plot_data(data, filename):
    plt.scatter(data[:, 0], data[:, 1])
    plt.title("PCA - 2D Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(filename)
    plt.show()

# Function to plot Likelihood values against K
def plot_likelihood(k_values, likelihood_values, filename):
    plt.subplot()
    plt.plot(k_values, likelihood_values, marker='o')
    plt.title("Log-Likelihood vs. K")
    plt.xlabel("Number of Components (K)")
    plt.ylabel("Log-Likelihood")
    plt.savefig(filename)
    plt.show()

# Function to plot estimated GMM for K'(best K)
def plot_gmm(best_k, data, filename):
    plt.subplot()
    gmm = GaussianMixtureModel(best_k)
    gmm.fit(data)
    labels = gmm.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(f"Estimated GMM for K = {best_k}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Implementation of GMM
class GaussianMixtureModel:
    def __init__(self, num_clusters, max_iterations=1000):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations

    def initialize(self, data):
        # Number of data points and features
        self.num_data_points, self.num_features = data.shape

        # Randomly initialize cluster means
        random_rows = np.random.randint(low=0, high=self.num_data_points, size=self.num_clusters)
        self.cluster_means = [data[row_index, :] for row_index in random_rows]

        # Initialize cluster covariances
        self.cluster_covariances = [np.cov(data.T) for _ in range(self.num_clusters)]

        # Initialize cluster priors
        self.cluster_priors = np.ones(self.num_clusters) / self.num_clusters

    def expectation_step(self, data):
        # Calculate the responsibilities (weights) of each cluster for each data point
        self.weights = self.calculate_probabilities(data)

        # Update cluster priors based on the mean weights
        self.cluster_priors = self.weights.mean(axis=0)

    def maximization_step(self, data):
        for cluster_index in range(self.num_clusters):
            # Update cluster means
            self.cluster_means[cluster_index] = self.weights[:, cluster_index].dot(data) / self.weights[:, cluster_index].sum()

            # Update cluster covariances
            self.cluster_covariances[cluster_index] = np.cov(data.T, aweights=(self.weights[:, cluster_index] / self.weights[:, cluster_index].sum()), bias=True)

            # Add a small regularization term to avoid singular covariances
            self.cluster_covariances[cluster_index] += np.identity(self.num_features) * 1e-8

    def fit(self, data):
        # Initialize model parameters
        self.initialize(data)
        self.log_likelihood = self.calculate_log_likelihood(data)

        # Run EM iterations until convergence or max iterations
        for iteration in range(self.max_iterations):
            # E-step: Expectation step
            self.expectation_step(data)

            # M-step: Maximization step
            self.maximization_step(data)

            # Calculate log likelihood and check for convergence
            current_log_likelihood = self.calculate_log_likelihood(data)
            if abs(current_log_likelihood - self.log_likelihood) < 1e-3:
                break
            self.log_likelihood = current_log_likelihood

    def calculate_probabilities(self, data):
        likelihoods = np.zeros((self.num_data_points, self.num_clusters))
        for cluster_index in range(self.num_clusters):
            # Multivariate normal distribution for each cluster
            distribution = multivariate_normal(
                mean=self.cluster_means[cluster_index],
                cov=self.cluster_covariances[cluster_index], allow_singular=True
            )
            likelihoods[:, cluster_index] = distribution.pdf(data)

        # Calculate weighted probabilities
        numerator = likelihoods * self.cluster_priors
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator

        return weights

    def calculate_log_likelihood(self, data):
        likelihoods = np.zeros((self.num_data_points, self.num_clusters))
        for cluster_index in range(self.num_clusters):
            distribution = multivariate_normal(
                mean=self.cluster_means[cluster_index],
                cov=self.cluster_covariances[cluster_index], allow_singular=True
            )
            likelihoods[:, cluster_index] = distribution.pdf(data)

        # Calculate log likelihood with a small constant to avoid log(0)
        log_likelihood = np.log(likelihoods.dot(self.cluster_priors) + 1e-5)
        return log_likelihood.sum()

    def predict(self, data):
        # Calculate the probability of each data point belonging to each cluster
        weights = self.calculate_probabilities(data)

        # Assign each data point to the cluster with the highest probability
        cluster_assignments = np.argmax(weights, axis=1)

        return cluster_assignments



# ---------- Main Program --------
if __name__ == "__main__":
    # Read the dataset
    dataset_filename = '3D_data_points.txt'
    data = read_dataset(dataset_filename)
    # print('data:\n',data)

    # Perform PCA
    print('Reducing dimension of data.....')
    data_2D = reduce_dimension(data)

    # print('2D data:\n',data_2D)

    # Plot PCA result
    plot_data(data_2D, 'data_2D.png')

    # Calculate log likelihood
    k_range = range(3, 9)

    print('Calculating log likelihood.......')
    likelihood_values = []
    for k in k_range:
      gmm = GaussianMixtureModel(k)
      gmm.fit(data_2D)
      likelihood_values.append([k,gmm.calculate_log_likelihood(data_2D)])

    likelihood_values = np.array(likelihood_values)
    # print(likelihood_values)
    # Plot log likelihood against K
    plot_likelihood(k_range, likelihood_values[:,1], 'log_likelihood.png')

    # Choose the best K as seen from Graph
    best_k = int(input('Enter best K(from Graph): '))

    # Plot GMM for the best value of K
    plot_gmm(best_k, data_2D, 'gmm_result.png')
    
    
    
