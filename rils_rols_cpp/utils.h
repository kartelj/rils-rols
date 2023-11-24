#pragma once
#include <vector>
#include "eigen/Eigen/Dense"

using namespace std;

class utils
{
public:
	static double R2(const Eigen::ArrayXd& y, const Eigen::ArrayXd& yp);

	static double RMSE(const Eigen::ArrayXd& y, const Eigen::ArrayXd& yp);
};

