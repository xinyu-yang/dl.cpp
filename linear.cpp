#include <cassert>
#include <iostream>
#include <stack>
#include <vector>
#include <cmath>
#include <random>

#include "layer.h"


class Linear : public Layer {
private:
    int inputSize_;
    int outputSize_;

    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;

	std::stack<std::vector<std::vector<double>>> stackOfInput_;

	int nonUpdateCount_;
	std::vector<std::vector<double>> lossW_;
	std::vector<double> lossB_;


public:
	/* The constructor consists of two parameters: the input size and
	 * the output size of this linear layer.
	 * We represent the input matrix as X, the weight matrix as W, the
	 * bias matrix as B. The output of linear part is Y.
	 */
    Linear(int inputSize, int outputSize) {
        inputSize_ = inputSize;
        outputSize_ = outputSize;

		/* Initialize the random engine from undeterministic random device */
        //std::random_device rd;
        //std::mt19937 gen(rd());
		std::default_random_engine gen(0.0); /* random engine with seed 0 */
        std::normal_distribution<double> dist(0.0, 1.0);

        /* Initialize the weight matrix with random values. The shape of weight matrix
		 * is (outputSize x inputSize), which is the same as pyTorch.
		 */
		std::vector<std::vector<double>> weights = {
			{-0.4249, -0.2123, 0.0295, -0.3806, 0.0869},
			{-0.0362, -0.0083, 0.2621, -0.0939, -0.2162},
			{-0.3615, 0.3067, -0.2667, -0.0698, -0.0887},
			{-0.1191, 0.1272, 0.1529, -0.3807, -0.0435}
		};
        weights_.resize(outputSize, std::vector<double>(inputSize));
        for (int r = 0; r < outputSize; ++r) {
            for (int c = 0; c < inputSize; ++c) {
                //weights_[r][c] = dist(gen);
                weights_[r][c] = weights[r][c];
            }
        }

        /* Initialize the bias vector with random values */
		std::vector<double> bias = {-0.0582, -0.4325, -0.3324, -0.1185};
        biases_.resize(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            //biases_[i] = dist(gen);
            biases_[i] = bias[i];
        }

		nonUpdateCount_ = 0;

		/* Initialize the derivative matrix of weights with respect to losses */
		lossW_.resize(outputSize, std::vector<double>(inputSize, 0));

		/* Initialize the derivative vector of bias with respect to losses */
		lossB_.resize(outputSize, 0);
    }


	/* The input is a batch of items, whose shape is (batchSize, inputSize).
	 * For each batch, computing Y = WX + B.
	 *
	 * The overall computation:
	 *		- multiply: inputSize * outputSize * batchSize
	 *		-      add: inputSize * outputSize * batchSize
	 *		-      add: outputSize * batchSize
	 * The overall memory:
	 *		-    input: inputSize * batchSize, created
	 *		-   output: outputSize * batchSize, created and returned
	 */
	std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) override {
		std::vector<std::vector<double>> output(input.size(), std::vector<double>(outputSize_, 0));
		stackOfInput_.push(input);

		double tmp = 0;
		std::vector<double> item;

		for (int batch = 0; batch < input.size(); batch++) {
			item = input[batch];
			assert(item.size() == inputSize_);
			for (int r = 0; r < outputSize_; r++) {
				for (int c = 0; c < inputSize_; c++) {
					tmp += weights_[r][c] * item[c];
				}
				output[batch][r] = tmp + biases_[r];
				tmp = 0;
			}
		}

		return output;
    }


	/* The input is the losses of outputs, whose size is (batchSize, outputSize)
	 * Generaly, the losses come from the next layer. Also, here we assume
	 * the order of BP is the same as FP. Thus, we use a stack to store the
	 * outputs of FP. This function does not work for other cases.
	 *
	 * For each batch, computing L(W) = L(Y)X^T, L(X) = L(Y)^TW. L(#) means the
	 * directive of original vector/matrix # with respect of loss.
	 *
	 * The overall computation:
	 *		- multiply: inputSize * outputSize * batchSize
	 *		- multiply: inputSize * outputSize * batchSize
	 *		-      add: inputSize * outputSize * batchSize
	 *		-      add: inputSize * outputSize * batchSize
	 *		-      add: outputSize * batchSize
	 * The overall memory:
	 *		-    input: batchSize * inputSize, poped out
	 *		-    lossW: inputSize * outputSize, used
	 *		-    lossB: outputSize, used
	 *		-    lossX: inputSize * batchSize, created and returned.
	 */
	std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& lossY) override {
		std::vector<std::vector<double>> lossX(lossY.size(), std::vector<double>(outputSize_, 0));
		if (stackOfInput_.empty()) {
			std::cerr << "The stack of inputs is empty!" << std::endl;
			exit(-1);
		}
		std::vector<std::vector<double>> X = stackOfInput_.top();
		if (X.size() != lossY.size()) {
			std::cerr << "The batch sizes of losses and inputs do not match!" << std::endl;
			exit(-1);

		}
		stackOfInput_.pop();

		for (int batch = 0; batch < lossY.size(); batch++) {
			assert(lossY[batch].size() == outputSize_);

			for (int r = 0; r < outputSize_; r++) {
				for (int c = 0; c < inputSize_; c++) {
					lossW_[r][c] += lossY[batch][r] * X[batch][c];
				}

				lossB_[r] += lossY[batch][r];
			}

			/* In these two loops, we reverse the computation order. So that we can
			 * access the weight matrix row by row.
			 */
			for (int r = 0; r < outputSize_; r++) {
				for (int c = 0; c < inputSize_; c++) {
					lossX[batch][c] += weights_[r][c] * lossY[batch][r];
				}
			}
		}

		return lossX;
    }

}; /* Linear class */


void printTwoDimensionalVector(const std::vector<std::vector<double>>& vec) {
    for (const auto& row : vec) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

void printOneDimensionalVector(std::vector<double>& vec) {
    for (const double& value : vec) {
		std::cout << value << " ";
    }
	std::cout << std::endl;
}

int main() {
    // Define the network architecture
    int inputSize = 5;
    int outputSize = 4;

    // Create an instance of the NeuralNetwork
    Linear network(inputSize, outputSize);

    // Train the neural network
	std::vector<std::vector<double>> inputs = {
    {1.0, 1.0, 1.0, 1.0, 1.0},
    {1.0, 1.0, 1.0, 1.0, 1.0}
	};

	// Perform forward and backward propagation
	std::vector<std::vector<double>> output = network.forward(inputs);
	printTwoDimensionalVector(output);
	//network.backPropagation(input, target);

    return 0;
}
