#include <vector>

class Layer {
public:
	virtual std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& x) = 0;
	virtual std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& x) = 0;
	virtual ~Layer() {}
};
