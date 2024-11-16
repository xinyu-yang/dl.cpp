#include <cmath>
#include <vector>


class Loss {
public:
	virtual double forward(double x) = 0;
	virtual double backward(double x) = 0;
	virtual ~Loss() {}
};

// Compute Mean Squared Error (MSE) Loss
double computeMSELoss(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Input sizes do not match.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double diff = predictions[i] - targets[i];
        sum += diff * diff;
    }

    return sum / predictions.size();
}

// Compute the derivative of MSE Loss w.r.t. predictions
std::vector<double> computeMSEDerivative(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Input sizes do not match.");
    }

    std::vector<double> derivatives(predictions.size(), 0.0);
    for (size_t i = 0; i < predictions.size(); ++i) {
        derivatives[i] = 2.0 * (predictions[i] - targets[i]) / predictions.size();
    }

    return derivatives;
}

// Compute Cosine Loss
double computeCosineLoss(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Input sizes do not match.");
    }

    double dotProduct = 0.0;
    double normPredictions = 0.0;
    double normTargets = 0.0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        dotProduct += predictions[i] * targets[i];
        normPredictions += predictions[i] * predictions[i];
        normTargets += targets[i] * targets[i];
    }

    double cosineSimilarity = dotProduct / (std::sqrt(normPredictions) * std::sqrt(normTargets));
    return 1.0 - cosineSimilarity;
}

// Compute the derivative of Cosine Loss w.r.t. predictions
std::vector<double> computeCosineDerivative(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Input sizes do not match.");
    }

    std::vector<double> derivatives(predictions.size(), 0.0);
    double normPredictions = 0.0;
    double normTargets = 0.0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        normPredictions += predictions[i] * predictions[i];
        normTargets += targets[i] * targets[i];
    }

    double normProduct = std::sqrt(normPredictions) * std::sqrt(normTargets);
    if (normProduct != 0.0) {
        for (size_t i = 0; i < predictions.size(); ++i) {
            derivatives[i] = (targets[i] / normProduct) - (predictions[i] * dotProduct / (normProduct * normProduct));
        }
    }

    return derivatives;
}

// Compute Cross-Entropy Loss
double computeCrossEntropyLoss(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Input sizes do not match.");
    }

    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss += -targets[i] * std::log(predictions[i]) - (1.0 - targets[i]) * std::log(1.0 - predictions[i]);
    }

    return loss / predictions.size();
}

// Compute the derivative of Cross-Entropy Loss w.r.t. predictions
std::vector<double> computeCrossEntropyDerivative(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Input sizes do not match.");
    }

    std::vector<double> derivatives(predictions.size(), 0.0);
    for (size_t i = 0; i < predictions.size(); ++i) {
        derivatives[i] = (predictions[i] - targets[i]) / (predictions[i] * (1.0 - predictions[i]));
    }

    return derivatives;
}
