# AI-Assisted Single-Neuron Binary Classifier

## Overview

This project was completed as part of **CS-530: AI Principles and Applications** at Southern New Hampshire University.

The project demonstrates how GitHub Copilot can be used as an AI-assisted development tool to translate functional requirements into working Python code, review the generated implementation, test the result, and verify that the final program satisfies the requirements.

The work was intentionally completed through an iterative AI-assisted process. GitHub Copilot generated and reviewed the code while I directed the prompts, evaluated the results, requested verification, and confirmed the required tests.

## What the Program Does

The program contains four main functions:

- **`neuron()`** calculates the output of a single artificial neuron.
- **`binary_classifier()`** applies the neuron to each row of a binary-classification data set.
- **`create_synthetic_data_set()`** creates a simple labeled data set with two input features.
- **`calculate_weights()`** trains the neuron weights for the selected activation function.

The neuron supports two activation functions:

- **Step**
- **Sigmoid**

## Verification

The final implementation was tested by:

1. Generating a synthetic data set.
2. Calculating weights using the step activation function.
3. Calculating weights using the sigmoid activation function.
4. Running the classifier with the step weights.
5. Running the classifier with the sigmoid weights.
6. Verifying that all required functions completed successfully.

Both activation-function paths produced matching binary predictions on the generated data set during the final test.

## Skills Demonstrated

- AI-assisted software development
- Prompt refinement
- Requirements interpretation
- Python
- Neural-network fundamentals
- Binary classification
- Step and sigmoid activation functions
- Testing and verification
- Human review of AI-generated code

## Responsible AI Use

This repository does **not** present the generated code as independently written professional software. It is an academic project demonstrating how an AI coding assistant can be directed, reviewed, tested, and verified by a human developer.

The project reinforced an important lesson from the course: generated code should be treated as a starting point that still requires human understanding, testing, and accountability.

## File

- [CS530_neuron.py](CS530_neuron.py)

## Author

Michael B. Wood  
B.S. Computer Science, Software Engineering concentration  
M.S. Artificial Intelligence student, Southern New Hampshire University
