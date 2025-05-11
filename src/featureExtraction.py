import pickle


def instructionTypesAsFeatures(pop, N):
    insHash = {}    # Dictionary to count occurrences of each instruction type across all individuals
    features = []

    # Loop through all individuals in the population
    for indiv in pop.individuals:
        # Loop through each instruction in the individual's sequence
        for ins in indiv.sequence:
            # Count the frequency of each instruction type across the whole population
            if ins.ins_type in insHash.keys():
                insHash[ins.ins_type] += 1
            else:
                insHash[ins.ins_type] = 1

    # Get a list of all unique instruction types (used as feature keys)
    allKeysList = list(insHash.keys())

    # Extract features for the top N individuals (most fit, assuming sorted strongest to weakest)
    for i in range(N):
        # Get the N-th best individual (reversed index due to sorting from strongest to weakest) - this was done for validation purposes thus it is optional.
        indiv = pop.getIndividual(N - 1 - i)

        # Initialize a list of zero counts for each instruction type
        allValues = [0] * len(allKeysList)

        # Count how many of each instruction type is used in this individual's sequence
        for ins in indiv.sequence:
            for key_ins in range(len(allKeysList)):
                if ins.ins_type == allKeysList[key_ins]:
                    allValues[key_ins] += ins.numOfInstructions

        # Get the fitness score of the individual (used as the first attribute)
        power = round(float(indiv.getFitness()), 6)

        # Build the feature vector: [fitness, feature1, feature2, ...]]
        indiv_features = []
        indiv_features.append(power)  # First attribute is the fitness score (e.g., power)
        indiv_features.extend(allValues)  # Followed by instruction counts

        # Add this individual's features to the dataset
        features.append(indiv_features)

    return features, allKeysList


def featureExtraction(file, N):
    # Open the binary file and load the population object using pickle
    input = open(file, "rb")
    pop = pickle.load(input)
    input.close()

    # Sort the individuals in the population by fitness (from best to worst)
    pop.sortByFitessToWeakest()

    # You can define any features you like by modifying the feature function.
    # To understand the expected return format, refer to the example shown below.
    feature_values, feature_names = instructionTypesAsFeatures(pop, N)


    # Example output:
    # return = (
    #     [[fitness_score1, feature1_value, feature2_value],   # Individual 1
    #      [fitness_score2, feature1_value, feature2_value],   # Individual 2
    #      [fitness_score3, feature1_value, feature2_value],   # Individual 3
    #      [...],
    #      [fitness_scoreN, feature1_value, feature2_value]],   # Individual N
    #     ["Feature1", "Feature2"]]  # Feature names
    # )
    # Where fitness score is the actual measurement (e.g., power consumption), and feature values is the value of each defined feature.
    # Return both the feature matrix and the list of instruction type keys
    return [feature_values, feature_names]
