#include "utils.h"
#include "eigen/Eigen/Dense"

double utils::R2(const Eigen::ArrayXd &y, const Eigen::ArrayXd &yp) {
	double y_avg = 0;
	for (auto yi : y)
		y_avg += yi;
	y_avg /= y.size();
	double ssr = 0, sst = 0;
	for (int i = 0; i < y.size(); i++) {
		ssr += pow(y[i] - yp[i], 2);
		sst += pow(y[i] - y_avg, 2);
	}
	return 1 - ssr / sst;
}

double utils::RMSE(const Eigen::ArrayXd& y, const Eigen::ArrayXd& yp) {
	double rmse = 0;
	for (int i = 0; i < y.size(); i++)
		rmse += pow(y[i] - yp[i], 2);
	return sqrt(rmse / y.size());
}