#include "TwoJointIK.h"
#include "BVH.h"
#include <functional>

TwoJointIK::
TwoJointIK(BVH* base_bvh, int idx)
	:mBVH(base_bvh)
{
	mOffsets = base_bvh->getOffsets();
	mParents = base_bvh->getParents();
	
	mRelatedNodeIndices.emplace_back(idx);
	mRelatedNodeIndices.emplace_back(mParents[idx]);
	mRelatedNodeIndices.emplace_back(mParents[mParents[idx]]);
}

void
TwoJointIK::
solve(const Eigen::Isometry3d& T_target,
		const Eigen::Vector3d& position,
		Eigen::MatrixXd& rotation, double eps)
{
	std::vector<Eigen::Isometry3d> Ts = mBVH->forwardKinematics(
		position,
		rotation,
		mRelatedNodeIndices[0]);

	Eigen::Vector3d a = Ts[2].translation();
	Eigen::Vector3d b = Ts[1].translation();
	Eigen::Vector3d c = Ts[0].translation();

	Eigen::Vector3d t = T_target.translation();
	Eigen::Quaterniond t_gr(T_target.linear());

	Eigen::Quaterniond a_lr(rotation.block<3,3>(0,3*mRelatedNodeIndices[2]));
	Eigen::Quaterniond b_lr(rotation.block<3,3>(0,3*mRelatedNodeIndices[1]));
	Eigen::Quaterniond a_gr(Ts[2].linear());

	auto length = [](const Eigen::Vector3d& x){ return x.norm(); };
	auto normalize = [](const Eigen::Vector3d& x){ return x.normalized(); };
	auto clip = [](double val,double min,double max){return std::min(std::max(val,min),max);};
	
	double lab = length(b - a);
	double lcb = length(b - c);
	double lat = clip(length(t-a),eps, lab + lcb - eps);

	double ac_ab_0 = std::acos(clip(normalize(c-a).dot(normalize(b-a)), -1.0, 1.0));
	double ba_bc_0 = std::acos(clip(normalize(a-b).dot(normalize(c-b)), -1.0, 1.0));
	double ac_ab_1 = std::acos(clip((lcb*lcb-lab*lab-lat*lat) / (-2*lab*lat), -1.0, 1.0));
	double ba_bc_1 = std::acos(clip((lat*lat-lab*lab-lcb*lcb) / (-2*lab*lcb), -1.0, 1.0));

	double ac_at_0 = std::acos(clip(normalize(c-a).dot(normalize(t-a)),-1.0, 1.0));

	Eigen::Vector3d axis0 = normalize((c-a).cross(b-a));
	Eigen::Vector3d axis1 = normalize((c-a).cross(t-a));
	
	axis0 = a_gr.inverse()*axis0;

	Eigen::Quaterniond r0(Eigen::AngleAxisd(ac_ab_1-ac_ab_0,axis0));
	Eigen::Quaterniond r1(Eigen::AngleAxisd(ba_bc_1-ba_bc_0,axis0));
	Eigen::Quaterniond r2(Eigen::AngleAxisd(ac_at_0,a_gr.inverse()*axis1));

	a_lr = a_lr * r0 * r2;
	b_lr = b_lr * r1;

	rotation.block<3,3>(0,3*mRelatedNodeIndices[2]) = a_lr.toRotationMatrix();
	rotation.block<3,3>(0,3*mRelatedNodeIndices[1]) = b_lr.toRotationMatrix();

	Ts = mBVH->forwardKinematics(
		position,
		rotation,
		mRelatedNodeIndices[0]);

	rotation.block<3,3>(0,3*mRelatedNodeIndices[0]) = rotation.block<3,3>(0,3*mRelatedNodeIndices[0])*Ts[0].linear().transpose()*t_gr;
}

// TwoJointIK::
// TwoJointIK(const SkeletonPtr& skel, BodyNode* bn)
// 	:mSkeleton(skel)
// {
// 	mRelatedBodyNodes.emplace_back(bn);
// 	mRelatedBodyNodes.emplace_back(bn->getParentBodyNode());
// 	mRelatedBodyNodes.emplace_back(bn->getParentBodyNode()->getParentBodyNode());

// 	for(int i=0;i<3;i++){
// 		mRelatedJoints.emplace_back(dynamic_cast<BallJoint*>(mRelatedBodyNodes[i]->getParentJoint()));
// 		mRelatedJointIndexInSkeletons.emplace_back(mRelatedJoints.back()->getIndexInSkeleton(0));
// 	}
// }

// Eigen::VectorXd
// TwoJointIK::
// solve(const Eigen::Isometry3d& T_target, bool apply, double eps)
// {
// 	std::vector<Eigen::Isometry3d> Ts;
// 	for(auto bn : mRelatedBodyNodes)
// 		Ts.emplace_back(bn->getTransform());

// 	Eigen::Vector3d a = Ts[2].translation();
// 	Eigen::Vector3d b = Ts[1].translation();
// 	Eigen::Vector3d c = Ts[0].translation();
// 	Eigen::Vector3d t = T_target.translation();
// 	Eigen::Quaterniond a_lr = dart::math::expToQuat(mRelatedJoints[2]->getPositions());
// 	Eigen::Quaterniond b_lr = dart::math::expToQuat(mRelatedJoints[1]->getPositions());
// 	Eigen::Quaterniond a_gr(Ts[2].linear());

// 	auto length = [](const Eigen::Vector3d& x){ return x.norm(); };
// 	auto normalize = [](const Eigen::Vector3d& x){ return x.normalized(); };
// 	auto clip = [](double val,double min,double max){return std::min(std::max(val,min),max);};
	
// 	double lab = length(b - a);
// 	double lcb = length(b - c);
// 	double lat = clip(length(t-a),eps, lab + lcb - eps);

// 	double ac_ab_0 = std::acos(clip(normalize(c-a).dot(normalize(b-a)), -1.0, 1.0));
// 	double ba_bc_0 = std::acos(clip(normalize(a-b).dot(normalize(c-b)), -1.0, 1.0));
// 	double ac_ab_1 = std::acos(clip((lcb*lcb-lab*lab-lat*lat) / (-2*lab*lat), -1.0, 1.0));
// 	double ba_bc_1 = std::acos(clip((lat*lat-lab*lab-lcb*lcb) / (-2*lab*lcb), -1.0, 1.0));

// 	double ac_at_0 = std::acos(clip(normalize(c-a).dot(normalize(t-a)),-1.0, 1.0));

// 	Eigen::Vector3d axis0 = normalize((c-a).cross(b-a));
// 	Eigen::Vector3d axis1 = normalize((c-a).cross(t-a));
	
// 	axis0 = a_gr.inverse()*axis0;

	
// 	Eigen::Quaterniond r0(Eigen::AngleAxisd(ac_ab_1-ac_ab_0,axis0));
// 	Eigen::Quaterniond r1(Eigen::AngleAxisd(ba_bc_1-ba_bc_0,axis0));
// 	Eigen::Quaterniond r2(Eigen::AngleAxisd(ac_at_0,a_gr.inverse()*axis1));

// 	a_lr = a_lr * r0 * r2;
// 	b_lr = b_lr * r1;

// 	Eigen::VectorXd new_pos = mSkeleton->getPositions();

	
// 	new_pos.segment<3>(mRelatedJointIndexInSkeletons[2]) = dart::math::quatToExp(a_lr);
// 	new_pos.segment<3>(mRelatedJointIndexInSkeletons[1]) = dart::math::quatToExp(b_lr);

// 	if(apply)
// 		mSkeleton->setPositions(new_pos);
// 	return new_pos;
// }