#ifndef __TWO_JOINT_IK_H__
#define __TWO_JOINT_IK_H__
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
class BVH;

class TwoJointIK
{
public:
	TwoJointIK(BVH* base_bvh, int heel_idx);
	void solve(const Eigen::Isometry3d& T_target,
			const Eigen::Vector3d& position,
			Eigen::MatrixXd& rotation, double eps = 1e-6);
private:
	BVH* mBVH;
	std::vector<Eigen::Vector3d> mOffsets;
	std::vector<int> mParents;

	std::vector<int> mRelatedNodeIndices;
};
#endif