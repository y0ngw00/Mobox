#include "Character.h"
#include "MathUtils.h"
#include "DARTUtils.h"
#include "MassSpringDamperSystem.h"
#include "Motion.h"
#include "BVH.h"
#include <algorithm>
#include <iostream>
using namespace dart;
using namespace dart::dynamics;
Character::
Character(dart::dynamics::SkeletonPtr& skel,
			const std::vector<dart::dynamics::BodyNode*>& end_effectors,
			const std::vector<std::string>& bvh_map,
			const Eigen::VectorXd& w_joint,
			const Eigen::VectorXd& kp,
			const Eigen::VectorXd& maxf)
	:mSkeleton(skel),mEndEffectors(end_effectors),mBVHMap(bvh_map), mJointWeights(w_joint),mKp(kp),mKv(2.0*kp.cwiseSqrt()),mMinForces(-maxf),mMaxForces(maxf)
{
	mAppliedForces = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
	mTargetPositions = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
}
void
Character::
addForceSensor(const Eigen::Vector3d& point)
{
	auto closest = DARTUtils::getPointClosestBodyNode(mSkeleton,point);
	mForceSensors.emplace_back(new ForceSensor(closest.first, closest.second));
}
void
Character::
setMSDParameters(const Eigen::VectorXd& mass_coeffs,
				const Eigen::VectorXd& spring_coeffs,
				const Eigen::VectorXd& damper_coeffs)
{
	mMassParams = mass_coeffs;
	mDamperParams = damper_coeffs;
	mSpringParams = spring_coeffs;
}
void
Character::
setBaseMotionAndCreateMSDSystem(Motion* m)
{
	//Reorganize parameter idx
	Eigen::VectorXd mass_coeffs(mSkeleton->getNumJoints());
	Eigen::VectorXd spring_coeffs(mSkeleton->getNumJoints());
	Eigen::VectorXd damper_coeffs(mSkeleton->getNumJoints());
	for(int i =0;i<mBVHIndices.size();i++)
	{
		mass_coeffs[mBVHIndices[i]] = mMassParams[i];
		spring_coeffs[mBVHIndices[i]] = mSpringParams[i];
		damper_coeffs[mBVHIndices[i]] = mDamperParams[i];
	}

	mForceSensors[0]->setInsomnia(true);
	mResponseDelay = 1;
	mMotion = m;
	mMSDMotion = new Motion();
	mMSDMotion->registerBVHHierarchy(mMotion->getBVH());
	mMSDSystem = new MassSpringDamperSystem(this,mass_coeffs,spring_coeffs,damper_coeffs,1.0/30.0);

	// mTwoJointIKs.emplace_back(new TwoJointIK(mMotion->getBVH(), mMotion->getBVH()->getNodeIndex("simLeftFoot")));
	// mTwoJointIKs.emplace_back(new TwoJointIK(mMotion->getBVH(), mMotion->getBVH()->getNodeIndex("simRightFoot")));
}
MassSpringDamperSystem*
Character::
getMSDSystem()
{
	return mMSDSystem;
}
void
Character::
stepMotion(const Eigen::VectorXd& action)
{
	int idx = mMotionStartFrame + mMotionCounter;
	int count = 0;
	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	for(int i =0;i<mForceSensors.size();i++)
	{
		auto fs = mForceSensors[i];
		if(!fs->isSleep())
		{
			// Eigen::VectorXd ai = action.segment(count*msd_dof,msd_dof);
			// mMSDSystem->addTargetPose(ai);
			// count++; //wrong
			Eigen::Vector3d ai = action;

			// Eigen::Vector3d ai = action.segment<3>(i*3);
			// XXXX = ai;
			// mMSDSystem->applyForce(fs->getBodyNode(), fs->getBodyNode()->getTransform().linear()*ai, fs->getLocalOffset());
			// mMSDSystem->applyForce(fs->getBodyNode(), ai, fs->getLocalOffset());
			mMSDSystem->applyForce(fs->getBodyNode(), fs->getBodyNode()->getTransform().linear()*(fs->getHapticPosition()*2000.0), fs->getLocalOffset());

			
		}

	}
	// XXXX = mMSDSystem->getTargetPose();
	auto pR_msd = mMSDSystem->step(mMotion->getPosition(idx), mMotion->getRotation(idx));
	// auto pR_msd = mMSDSystem->stepPose(mMotion->getPosition(idx), mMotion->getRotation(idx));
	Eigen::Vector3d p_msd = pR_msd.first;
	Eigen::MatrixXd R_msd = pR_msd.second;
	mMSDMotion->append(p_msd, R_msd);
	// Eigen::Vector3d p_root_diff = p_msd - mMotion->getPosition(idx);
	// int lf = mMotion->getBVH()->getNodeIndex("simLeftFoot");
	// int rf = mMotion->getBVH()->getNodeIndex("simRightFoot");
	// Eigen::Isometry3d Tlf = mMotion->getBVH()->forwardKinematics(p_msd, R_msd, lf)[0];
	// Eigen::Isometry3d Trf = mMotion->getBVH()->forwardKinematics(p_msd, R_msd, rf)[0];
	// Tlf.translation() -= p_root_diff;
	// Trf.translation() -= p_root_diff;
	// Eigen::Isometry3d T_target = Ts[0];
	// T_target.translation()[1] += 0.4;
	// // T_target.translation()[0] += 0.1;
	// mTwoJointIKs[0]->solve(Tlf, p_msd, R_msd);
	// mTwoJointIKs[1]->solve(Trf, p_msd, R_msd);
	
	mMotionCounter++;
}
void
Character::
resetMotion(int start_frame)
{
	mMotionStartFrame = start_frame;
	
	mMSDSystem->reset();
	mMSDMotion->clear();
	int idx = start_frame;
	for(int i=0;i<mResponseDelay;i++)
	{
		mMSDMotion->append(mMotion->getPosition(idx),mMotion->getRotation(idx),false);
		idx++;
	}
	mMSDMotion->append(mMotion->getPosition(idx), mMotion->getRotation(idx));

	mMotionCounter = mResponseDelay;

}
const Eigen::Vector3d&
Character::
getPosition(int idx)
{
	return mMSDMotion->getPosition(idx - mMotionStartFrame);
}
const Eigen::MatrixXd&
Character::
getRotation(int idx)
{
	return mMSDMotion->getRotation(idx - mMotionStartFrame);
}
const Eigen::Vector3d&
Character::
getLinearVelocity(int idx)
{
	return mMSDMotion->getLinearVelocity(idx - mMotionStartFrame);
}
const Eigen::MatrixXd&
Character::
getAngularVelocity(int idx)
{
	return mMSDMotion->getAngularVelocity(idx - mMotionStartFrame);
}
Eigen::Isometry3d
Character::
getRootTransform()
{
	Eigen::VectorXd pos = mSkeleton->getPositions();
	return FreeJoint::convertToTransform(pos.head<6>());
}
void
Character::
setRootTransform(const Eigen::Isometry3d& T)
{
	Eigen::VectorXd pos = mSkeleton->getPositions();
	pos.head<6>() = FreeJoint::convertToPositions(T);

	mSkeleton->setPositions(pos);
}

Eigen::Isometry3d
Character::
getReferenceTransform()
{
	Eigen::Isometry3d T = this->getRootTransform();
	Eigen::Matrix3d R = T.linear();
	Eigen::Vector3d p = T.translation();
	Eigen::Vector3d z = R.col(2);
	Eigen::Vector3d y = Eigen::Vector3d::UnitY();
	z -= MathUtils::projectOnVector(z, y);
	p -= MathUtils::projectOnVector(p, y);

	z.normalize();
	Eigen::Vector3d x = y.cross(z);

	R.col(0) = x;
	R.col(1) = y;
	R.col(2) = z;

	T.linear() = R;
	T.translation() = p;

	return T;
}
void
Character::
setReferenceTransform(const Eigen::Isometry3d& T_ref)
{
	Eigen::Isometry3d T_ref_cur = this->getReferenceTransform();
	Eigen::Isometry3d T_cur = this->getRootTransform();
	Eigen::Isometry3d T_diff = T_ref_cur.inverse()*T_cur;
	Eigen::Isometry3d T_next = T_ref*T_diff;

	this->setRootTransform(T_next);
}
void
Character::
setPose(const Eigen::Vector3d& position,
		const Eigen::MatrixXd& rotation)
{
	Eigen::VectorXd p = this->toSimPose(position,rotation);
	Eigen::VectorXd v = Eigen::VectorXd::Zero(p.rows());

	mSkeleton->setPositions(p);
	mSkeleton->setVelocities(v);
	mSkeleton->computeForwardKinematics(true,true,false);
}
void
Character::
setPose(const Eigen::Vector3d& position,
		const Eigen::MatrixXd& rotation,
		const Eigen::Vector3d& linear_velocity,
		const Eigen::MatrixXd& angular_velocity)
{
	Eigen::VectorXd p = this->toSimPose(position,rotation);
	Eigen::VectorXd v = this->toSimVel(position,rotation, linear_velocity, angular_velocity);

	mSkeleton->setPositions(p);
	mSkeleton->setVelocities(v);
	mSkeleton->computeForwardKinematics(true,true,false);
}
std::pair<Eigen::VectorXd,Eigen::VectorXd>
Character::
computeTargetPosAndVel(const Eigen::MatrixXd& base_rot, const Eigen::VectorXd& action,
					const Eigen::MatrixXd& angular_velocity)
{
	Eigen::VectorXd target_position = this->toSimPose(Eigen::Vector3d::Zero(),base_rot);
	Eigen::VectorXd target_velocity = this->toSimVel(Eigen::Vector3d::Zero(),base_rot, Eigen::Vector3d::Zero(), angular_velocity);
	int cnt = 0;
	for(int i=0;i<mSkeleton->getNumJoints();i++)
	{
		if(mSkeleton->getJoint(i)->getType()=="BallJoint")
		{
			int idx_in_skel = mSkeleton->getJoint(i)->getIndexInSkeleton(0);
			
			Eigen::Matrix3d Ri = BallJoint::convertToRotation(target_position.segment<3>(idx_in_skel));
			Eigen::Matrix3d dR = BallJoint::convertToRotation(action.segment<3>(3*cnt));

			target_position.segment<3>(idx_in_skel) = BallJoint::convertToPositions(Ri*dR);
			// target_position.segment<3>(idx_in_skel) = BallJoint::convertToPositions(dR);
			// target_velocity.segment<3>(idx_in_skel) += mMotion->getTimestep()*action.segment<3>(3*cnt);
			cnt++;
		}
	}
	// target_position.tail(target_position.rows()-6) += action;
	mTargetPositions = target_position;
	return std::make_pair(target_position, target_velocity);
}

void
Character::
actuate(const Eigen::VectorXd& target_position, const Eigen::VectorXd& target_velocity)
{
	// Eigen::VectorXd target_position = this->toSimPose(Eigen::Vector3d::Zero(),target_rotation);

	Eigen::VectorXd q = mSkeleton->getPositions();
	Eigen::VectorXd dq = mSkeleton->getVelocities();
	double dt = mSkeleton->getTimeStep();

	Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();

	Eigen::VectorXd qdqdt = q + dq*dt;

	Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt,target_position));
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq - target_velocity);
	Eigen::VectorXd ddq = M_inv*(-mSkeleton->getCoriolisAndGravityForces()+p_diff+v_diff+mSkeleton->getConstraintForces());

	Eigen::VectorXd tau = p_diff + v_diff - dt*mKv.cwiseProduct(ddq);

	tau = dart::math::clip<Eigen::VectorXd,Eigen::VectorXd>(tau,mMinForces,mMaxForces);
	mAppliedForces = tau;

	mSkeleton->setForces(tau);
}
std::map<std::string, Eigen::MatrixXd>
Character::
getStateBody()
{
	int n = mSkeleton->getNumBodyNodes();
	Eigen::MatrixXd ps(3,n), Rs(3,3*n), vs(3,n), ws(3,n);

	for(int i=0;i<n;i++)
	{
		Eigen::Isometry3d Ti = mSkeleton->getBodyNode(i)->getTransform();
		ps.col(i) = Ti.translation();
		Rs.block<3,3>(0,i*3) = Ti.linear();
		vs.col(i) = mSkeleton->getBodyNode(i)->getLinearVelocity();
		ws.col(i) = mSkeleton->getBodyNode(i)->getAngularVelocity();
	}
	std::map<std::string, Eigen::MatrixXd> state;

	state.insert(std::make_pair("ps",ps));
	state.insert(std::make_pair("Rs",Rs));
	state.insert(std::make_pair("vs",vs));
	state.insert(std::make_pair("ws",ws));

	return state;
}
std::map<std::string, Eigen::MatrixXd>
Character::
getStateJoint()
{
	int n = mSkeleton->getNumJoints();
	Eigen::MatrixXd p(3,n), v(3,n);
	for(int i = 0;i<n;i++)
	{
		auto joint = mSkeleton->getJoint(i);
		if(joint->getType() == "FreeJoint"){
			p.col(i) = joint->getPositions().head<3>();
			v.col(i) = joint->getVelocities().head<3>();
		}
		else if(joint->getType() == "BallJoint"){
			p.col(i) = joint->getPositions();
			v.col(i) = joint->getVelocities();
		}
		else{
			p.col(i) = Eigen::Vector3d::Zero();
			v.col(i) = Eigen::Vector3d::Zero();
		}
	}
	std::map<std::string, Eigen::MatrixXd> state;

	state.insert(std::make_pair("p",p));
	state.insert(std::make_pair("v",v));

	return state;
}
std::map<std::string, Eigen::MatrixXd>
Character::
getStateForceSensors()
{
	Eigen::Isometry3d T_ref = this->getReferenceTransform();

	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	int n = mForceSensors.size();

	std::vector<Eigen::Vector3d> ps, vs, hps, hvs;
	for(int i=0;i<n;i++)
	{
		ps.emplace_back(T_ref_inv*mForceSensors[i]->getPosition());
		vs.emplace_back(R_ref_inv*mForceSensors[i]->getVelocity());
		hps.emplace_back(mForceSensors[i]->getHapticPosition());
		hvs.emplace_back(mForceSensors[i]->getHapticVelocity());
	}
	int m = ps.size();
	Eigen::MatrixXd pse(3,m), vse(3,m), hpse(3,m), hvse(3,m);
	for(int i=0;i<m;i++)
	{
		pse.col(i) = ps[i];
		vse.col(i) = vs[i];
		hpse.col(i) = hps[i];
		hvse.col(i) = hvs[i];
	}
	std::map<std::string, Eigen::MatrixXd> state;

	state.insert(std::make_pair("ps",pse));
	state.insert(std::make_pair("vs",vse));
	state.insert(std::make_pair("hps",hpse));
	state.insert(std::make_pair("hvs",hvse));

	return state;
}
std::vector<Eigen::Vector3d>
Character::
getState()
{
	Eigen::Isometry3d T_ref = this->getReferenceTransform();

	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	int n = mSkeleton->getNumBodyNodes();
	std::vector<Eigen::Vector3d> ps(n),vs(n),ws(n);
	std::vector<Eigen::MatrixXd> Rs(n);

	for(int i=0;i<n;i++)
	{
		Eigen::Isometry3d Ti = T_ref_inv*(mSkeleton->getBodyNode(i)->getTransform());

		ps[i] = Ti.translation();
		Rs[i] = Ti.linear();

		vs[i] = R_ref_inv*mSkeleton->getBodyNode(i)->getLinearVelocity();
		ws[i] = R_ref_inv*mSkeleton->getBodyNode(i)->getAngularVelocity();
	}
	Eigen::Vector3d p_com = T_ref_inv*mSkeleton->getCOM();
	Eigen::Vector3d v_com = R_ref_inv*mSkeleton->getCOMLinearVelocity();

	std::vector<Eigen::Vector3d> states(5*n+2);

	int o = 0;
	for(int i=0;i<n;i++) states[o+i] = ps[i]; o += n;
	for(int i=0;i<n;i++) states[o+i] = Rs[i].col(0); o += n;
	for(int i=0;i<n;i++) states[o+i] = Rs[i].col(1); o += n;
	for(int i=0;i<n;i++) states[o+i] = vs[i]; o += n;
	for(int i=0;i<n;i++) states[o+i] = ws[i]; o += n;

	states[o+0] = p_com;
	states[o+1] = v_com;

	return states;
}
// std::vector<Eigen::Vector3d>
// Character::
// getStateForceSensor()
// {
// 	Eigen::Isometry3d T_ref = this->getReferenceTransform();

// 	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
// 	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

// 	int n = mForceSensors.size();
	
// 	std::vector<Eigen::Vector3d> ps(n), vs(n);
// 	states(n*m);
// 	for(int i=0;i<n;i++)
// 	{
// 		auto fs = mForceSensors[i];
// 		ps[i] = T_ref_inv*(fs->getPosition());
// 		vs[i] = R_ref_inv*(fs->body_node->getLinearVelocity(fs->local_pos));
// 			forces[i*m+j] = R_ref_inv*fs->value[j];
// 		}
// 	}

// 	std::vector<Eigen::Vector3d> states((2+m)*n);

// 	int o = 0;
// 	for(int i=0;i<n;i++) states[o+i] = ps[i]; o += n;
// 	for(int i=0;i<n;i++) states[o+i] = vs[i]; o += n;
// 	for(int i=0;i<n*m;i++) states[o+i] = forces[i];

// 	return states;
// }

Eigen::VectorXd
Character::
saveState()
{
	Eigen::VectorXd p = mSkeleton->getPositions();
	Eigen::VectorXd v = mSkeleton->getVelocities();

	Eigen::VectorXd state(p.rows()+v.rows());
	state<<p,v;

	return state;
}

void
Character::
restoreState(const Eigen::VectorXd& state)
{
	int o = 0;
	int n = mSkeleton->getNumDofs();
	Eigen::VectorXd p = state.segment(o,n);
	o += n;
	Eigen::VectorXd v = state.segment(o,n);

	mSkeleton->setPositions(p);
	mSkeleton->setVelocities(v);
}
void
Character::
buildBVHIndices(const std::vector<std::string>& bvh_names)
{
	mBVHIndices.reserve(mBVHMap.size());

	for(int i=0;i<mBVHMap.size();i++)
	{
		int index = std::distance(bvh_names.begin(),std::find(bvh_names.begin(),bvh_names.end(), mBVHMap[i]));
		if(index>=bvh_names.size())
			mBVHIndices.emplace_back(-1);
		else
			mBVHIndices.emplace_back(index);
	}
}
Eigen::VectorXd
Character::
toSimPose(const Eigen::Vector3d& position, const Eigen::MatrixXd& rotation)
{
	assert(mBVHIndices.size()!=0);
	Eigen::VectorXd p = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
	p.segment<3>(3) = position;
	for(int i=0;i<mBVHIndices.size();i++)
	{
		int idx = mBVHIndices[i];
		if(mSkeleton->getJoint(i)->getNumDofs()==0 || idx<0)
			continue;
		Eigen::Matrix3d R = rotation.block<3,3>(0,idx*3);
		int idx_in_skel = mSkeleton->getJoint(i)->getIndexInSkeleton(0);
		p.segment<3>(idx_in_skel) = BallJoint::convertToPositions(R);
	}
	return p;
}
Eigen::VectorXd
Character::
toSimVel(const Eigen::Vector3d& position, const Eigen::MatrixXd& rotation, const Eigen::Vector3d& linear_velocity, const Eigen::MatrixXd& angular_velcity)
{
	assert(mBVHIndices.size()!=0);
	Eigen::VectorXd v = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
	

	for(int i=0;i<mBVHIndices.size();i++)
	{
		int idx = mBVHIndices[i];
		if(mSkeleton->getJoint(i)->getNumDofs()==0 || idx<0)
			continue;

		int idx_in_skel = mSkeleton->getJoint(i)->getIndexInSkeleton(0);
		v.segment<3>(idx_in_skel) = angular_velcity.col(idx);
		if (idx_in_skel==0)
		{
			Eigen::Matrix3d R_root = rotation.block<3,3>(0,0);
			// v.segment<3>(3) = linear_velocity;
			v.segment<3>(3) = R_root.transpose()*linear_velocity;
		}
	}
	
	return v;
}

ForceSensor*
Character::
getClosestForceSensor(const Eigen::Vector3d& point)
{
	double min_distance =1e6;
	ForceSensor* min_fs = nullptr;
	for(auto fs : mForceSensors)
	{
		Eigen::Vector3d p_glob = fs->getPosition();
		double distance = (p_glob-point).norm();
		if(distance<min_distance)
		{
			min_distance = distance;
			min_fs = fs;
		}
	}

	if(min_distance<0.1)
		return min_fs;
	return nullptr;
}
std::pair<ForceSensor*,double>
Character::
getClosestForceSensor(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1)
{
	double min_distance =1e6;
	ForceSensor* min_fs = nullptr;
	double min_t = 0;
	for(auto fs : mForceSensors)
	{
		Eigen::Vector3d p_glob = fs->getPosition();
		double t = (p1 - p0).dot(p_glob - p0)/(p1 - p0).dot(p1 - p0);
		double distance = (t*(p1-p0) - (p_glob - p0)).norm();
		if(distance<min_distance)
		{
			min_distance = distance;
			min_fs = fs;
			min_t = t;
		}
	}

	return std::make_pair(min_fs, min_t);
}
