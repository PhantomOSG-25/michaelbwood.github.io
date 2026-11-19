# CS 530: Single-Neuron Binary Classifier


def neuron(inputs, weights, activation="step"):
    """
    Header: Calculate a neuron output using a selected activation function.

    Args:
        inputs: A list of numeric input features.
        weights: A list containing one weight per input, followed by a bias
            as the final item.
        activation: "step" or "sigmoid".

    Returns:
        The activated neuron output. Step returns 0 or 1; sigmoid returns
        a value between 0 and 1.
    """
    if activation not in ("step", "sigmoid"):
        raise ValueError("activation must be 'step' or 'sigmoid'")

    if len(weights) != len(inputs) + 1:
        raise ValueError("weights must contain one weight per input plus one bias")

    weighted_sum = 0.0
    for index, value in enumerate(inputs):
        weighted_sum += value * weights[index]
    weighted_sum += weights[-1]

    if activation == "step":
        return 1 if weighted_sum >= 0 else 0

    if weighted_sum >= 0:
        sigmoid_value = 1.0 / (1.0 + (2.718281828459045 ** -weighted_sum))
    else:
        exponential = 2.718281828459045 ** weighted_sum
        sigmoid_value = exponential / (1.0 + exponential)
    return sigmoid_value


def binary_classifier(data_set, weights, activation="step"):
    """
    Header: Apply a neuron to each row of a binary-classification data set.

    Args:
        data_set: A list of rows containing input features and a label.
        weights: A list containing feature weights followed by a bias.
        activation: "step" or "sigmoid".

    Returns:
        A list of predictions corresponding to each data-set row.
    """
    predictions = []
    for row in data_set:
        if len(row) < 2:
            raise ValueError("each data-set row must contain features and a label")

        inputs = row[:-1]
        prediction = neuron(inputs, weights, activation)
        if activation == "sigmoid":
            prediction = 1 if prediction >= 0.5 else 0
        predictions.append(prediction)

    return predictions


def create_synthetic_data_set(number_of_rows=20):
    """
    Header: Create a simple two-feature synthetic binary-classification data set.

    Args:
        number_of_rows: The number of rows to generate.

    Returns:
        A list of rows formatted as [feature1, feature2, label].
    """
    if number_of_rows < 1:
        raise ValueError("number_of_rows must be at least 1")

    data_set = []
    for index in range(number_of_rows):
        feature1 = float(index)
        feature2 = float((index * 3) % (number_of_rows + 1))
        label = 1 if feature1 + feature2 >= number_of_rows / 2 else 0
        data_set.append([feature1, feature2, label])

    return data_set


def calculate_weights(activation="step", data_set=None):
    """
    Header: Calculate neuron weights from a binary-classification data set.

    Args:
        activation: "step" or "sigmoid".
        data_set: A list of rows containing input features and a label.

    Returns:
        A list containing learned feature weights followed by a bias.
    """
    if activation not in ("step", "sigmoid"):
        raise ValueError("activation must be 'step' or 'sigmoid'")
    if not data_set:
        raise ValueError("data_set must contain at least one row")

    feature_count = len(data_set[0]) - 1
    weights = [0.0] * (feature_count + 1)
    learning_rate = 0.01

    for epoch in range(1000):
        total_error = 0.0

        for row in data_set:
            if len(row) != feature_count + 1:
                raise ValueError("all data-set rows must have the same number of features")

            inputs = row[:-1]
            label = row[-1]
            if label not in (0, 1):
                raise ValueError("labels must be either 0 or 1")

            raw_output = 0.0
            for index, value in enumerate(inputs):
                raw_output += value * weights[index]
            raw_output += weights[-1]

            if activation == "step":
                prediction = 1 if raw_output >= 0 else 0
                output = float(prediction)
            else:
                if raw_output >= 0:
                    output = 1.0 / (1.0 + (2.718281828459045 ** -raw_output))
                else:
                    exponential = 2.718281828459045 ** raw_output
                    output = exponential / (1.0 + exponential)

            error = label - output
            total_error += abs(error)

            for index, value in enumerate(inputs):
                weights[index] += learning_rate * error * value
            weights[-1] += learning_rate * error

        if total_error == 0:
            break

    return weights
