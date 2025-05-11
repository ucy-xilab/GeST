import numpy as np
from scipy.optimize import curve_fit


# Compute Mean Squared Error.
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def log_function(x, a, b):
    return a + b * np.log10(x)


def log_function_2(x, a, b):
    return a + b * np.log(x)


def log_function_3(x, a, b):
    return a + b * np.log2(x)


def getReferenceFeatures(feature_values_names):
    # Example input:
    # feature_values_names = (
    #     [[fitness_score1, feature1_value, feature2_value],   # Individual 1
    #      [fitness_score2, feature1_value, feature2_value],   # Individual 2
    #      [fitness_score3, feature1_value, feature2_value],   # Individual 3
    #      [...],
    #      [fitness_scoreN, feature1_value, feature2_value]],   # Individual N
    #     ["Feature1", "Feature2"]]  # Feature names
    # )

    # Unpack input into feature_values and feature_names
    feature_values = feature_values_names[0]  # 2D list of data points
    feature_names = feature_values_names[1]   # List of feature names

    # Create a list of empty lists, one for each feature/column
    # In this case: features = [[], [], []] to hold values for A, B, C
    features = [[] for _ in range(len(feature_values[0]))]


    # Transpose the data: group values by feature instead of by row
    for i in range(len(features)):           # For each feature/column (i = 0 to 2)
        for j in range(len(feature_values)): # For each row (j = 0 to 2)
            # Append the value at row j, column i into features[i]
            features[i].append(feature_values[j][i])

    # After processing:
    # features = [
    #     [1, 4, 7, ...],  # Feature1 values
    #     [2, 5, 8, ...],  # Feature2 values
    # ]

    # Convert the first attribute (e.g., power consumption) into an array (it is the fitness_Score)
    # This will act as the independent variable (X-axis) for curve fitting
    X = np.array(features[0])

    # This will store predictions for each feature (e.g., instruction type)
    reference_features = []

    # List of candidate functions to fit the data with
    candidate_functions = [log_function, log_function_2, log_function_3]

    # Loop over all features except the first one (which is X: power consumption)
    for i in range(1, len(features)):

        # Get the actual Y values for this instruction type
        Y = np.array(features[i])
        best_params = None  # Will hold the best-fitting parameters
        best_mse = float('inf')  # Track the lowest mean squared error found
        best_function = None  # Store the function that gives best fit

        # Try fitting each candidate function to the data
        for func in candidate_functions:
            try:
                # Fit the function to the data (curve fitting)
                params, covariance = curve_fit(func, X, Y)

                # Calculate Mean Squared Error (MSE) between actual and predicted Y values
                mse_fit = compute_mse(Y, func(X, *params))

                # Update best function if this one is better
                if mse_fit < best_mse:
                    best_params = params
                    best_mse = mse_fit
                    best_function = func

            except Exception as e:
                # If the curve fitting fails (e.g., bad initial guess), log the error
                print(f"Error fitting {func.__name__}: {e}")

        # This simulates prediction at higher power/ipc/etc.
        X_max = np.max(X)
        offset = 100

        if best_params is not None:

            # This example applies to power-virus generation:
            # The power target for feature prediction is set to X_max + offset,
            # where X_max is the maximum observed power in the latest generation,
            # and the offset (e.g., 100W) is empirically chosen.
            # From our experience, this extrapolated offset helps the surrogate function (SF) emphasize
            # feature contribution to power and improves ranking effectiveness.
            # While the target may exceed realistic power limits, it was found to improve efficiency.
            # You may adjust the offset as needed for your platform or experiment,
            # or experiment with a completely different target (e.g., a fixed value).

            # Predict using the best function and parameters.
            ypred = best_function(X_max + offset, *best_params)

            # Avoid negative predictions (which might not make sense for things like instruction counts)
            if ypred < 0:
                ypred = 0

            # Print out some results for features, for validation purposes
            # print(f"Best Function: {best_function.__name__}")
            # print(f"Best Parameters: {best_params}")
            # print(f"Prediction: {ypred}")
            # print(f"MSE: {best_mse}")

        else:
            # If all fits failed, default to zero prediction
            ypred = 0

            # Save the predicted value
        reference_features.append([ypred])

        # Return predictions and the corresponding feature names
    return [reference_features, feature_names]
