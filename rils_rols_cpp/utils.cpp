#include "utils.h"
#include "eigen/Eigen/Dense"

double utils::R2(const Eigen::ArrayXd &y, const Eigen::ArrayXd &yp) {
	const auto y_avg = y.mean();
	const auto ssr = ((y - yp) * (y - yp)).sum();
	const auto sst = ((y - y_avg) * (y - y_avg)).sum();
	return 1 - ssr / sst;
}

double utils::RMSE(const Eigen::ArrayXd& y, const Eigen::ArrayXd& yp) {
	return sqrt(((y - yp) * (y - yp)).mean());
}

double utils::classification_accuracy(const Eigen::ArrayXd& y, const Eigen::ArrayXd& yp) {
	double acc = 0;
	for (int i = 0; i < y.size(); i++) {
		// binarized value
		const auto ypib = yp[i] >= 0.5 ? 1.0 : 0.0;
		const auto yib = y[i] >= 0.5 ? 1.0 : 0.0;
		if (ypib == yib)
			acc += 1.0;
	}
	return acc / y.size();
}

double utils::average_log_loss(const Eigen::ArrayXd& y, const Eigen::ArrayXd& yp)
{
	double ll = 0;
	for (int i = 0; i < y.size(); i++) {
		const auto yib = y[i] >= 0.5 ? 1.0 : 0.0;
		//double ypi = yp[i];
		//if (ypi != 0)
		//	ypi = ypi;
		const auto prob = 1.0 / (1.0 + exp(-2.0 * (yp[i] - 0.5))); // logistic function for mean at 0.5
		const auto lli = (1.0 - yib) * log(1.0 - prob) + yib * log(prob);
		ll -= lli;
	}
	return ll / y.size();
}

double utils::average_loss(const Eigen::ArrayXd& y, const Eigen::ArrayXd& yp)
{
	double ll = 0;
	for (int i = 0; i < y.size(); i++) {
		const auto yib = y[i] >= 0.5 ? 1.0 : 0.0;
		ll+= abs(yib - yp[i]);
	}
	return ll / y.size();
}
