import numpy as np

def measureMeasurementNoise(instance):
    # instance is 'self' from the calling class
    individuals_to_check = 30
    individual_measurements = 10
    individual_cvs = []

    for individual in instance.population.individuals[:individuals_to_check]:
        individual_measurements_array = []
        for i in range(individual_measurements):
            while True:
                try:
                    measurements = instance.__measureIndividual__(individual)
                    measurement = measurements[0]
                    individual_measurements_array.append(float(measurement))
                    print(measurement)
                    break
                except (ValueError, IOError):
                    continue

        individual.setMeasurementsVector(measurements)
        print(individual_measurements_array)
        mean = np.mean(individual_measurements_array)
        std = np.std(individual_measurements_array)
        cv = std / mean if mean else 0
        print(f"Individual's measurements mean:{mean}, std:{std}, cv:{cv}")
        individual_cvs.append(cv)

    mean_cv = np.mean(individual_cvs)
    std_cv = np.std(individual_cvs)
    platform_cv = std_cv / mean_cv if mean_cv else 0

    print(f"\nPlatform Measurement CV (CV of CVs): {platform_cv:.4f}")
    if platform_cv > 0.65:
        instance.N = 20
        print("High Noise Detected... N = 20")
    else:
        instance.N = 10
        print("Low Noise Detected... N = 10")