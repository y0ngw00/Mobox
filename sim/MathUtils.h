#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
class MathUtils
{
public:

	static Eigen::Vector3d projectOnVector(const Eigen::Vector3d& u, const Eigen::Vector3d& v);
	static Eigen::Isometry3d orthonormalize(const Eigen::Isometry3d& T_old);
	static Eigen::VectorXd ravel(const std::vector<Eigen::Vector3d>& vv);
	static Eigen::VectorXd ravel(const std::vector<Eigen::VectorXd>& vv);
};
#endif