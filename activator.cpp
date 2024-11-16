#include <cctype>
#include <iostream>
#include <cmath>

class Activation {
public:
	virtual double forward(double x) = 0;
	virtual double backward(double x) = 0;
	virtual ~Activation() {}
};


class ReLU : public Activation {
public:
	double forward(double x) override {
        return std::max(0.0, x);
    }

    double backward(double x) override {
        return x > 0 ? 1 : 0;
    }
};


class Sigmoid : public Activation {
public:
	double forward(double x) override {
        return 1 / (1 + exp(-x));
    }

    double backward(double x) override {
        return x * (1 - x);
    }
};


class Tanh : public Activation {
public:
	double forward(double x) override {
        return std::tanh(x);
    }

    double backward(double x) override {
        return 1 - (x * x);
    }
};


class Activator {
private:
	Activation * activation;


public:
	Activator(std::string& activationName) {
		if (activationName.empty()) {
			activationName = "ReLU";
		}

		// To upper case
		for (char& c : activationName) {
			c = std::tolower(c);
		}

		switch (activationName[0]) {
			case 's':
				activation = new Sigmoid();
				break;
			case 'r':
				activation = new ReLU();
				break;
			case 't':
				activation = new Tanh();
				break;
			default:
				std::cerr << "Invalid activation function specified!" << std::endl;
				activation = new ReLU();
				break;
		}
	}

	~Activator() {
		delete activation;
	}

	double forward(double x) {
		return activation->forward(x);
	}

	double backward(double x) {
		return activation->backward(x);
	}
};
