#include "Character.h"
#include "MathUtils.h"
#include "DARTUtils.h"
#include "Motion.h"
#include "BVH.h"
#include <algorithm>
#include <iostream>
#include <sstream>
using namespace dart;
using namespace dart::dynamics;

Character::
Character(dart::dynamics::SkeletonPtr& skel,
			const std::vector<dart::dynamics::BodyNode*>& end_effectors,
			const std::vector<std::string>& bvh_map,
			const Eigen::VectorXd& w_joint,
			const Eigen::VectorXd& kp,
			const Eigen::VectorXd& maxf)
	:mSkeleton(skel),
	mEndEffectors(end_effectors),
	mBVHMap(bvh_map),
	mJointWeights(w_joint),
	mKp(kp),
	mKv(2.0*kp.cwiseSqrt()),
	mMinForces(-maxf),
	mMaxForces(maxf),
	mTargetPositions(Eigen::VectorXd::Zero(skel->getNumDofs()))
{
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
Eigen::VectorXd
Character::
computeTargetPosition(const Eigen::VectorXd& action)
{
	int n = mSkeleton->getNumDofs();
	mTargetPositions.tail(n-6) = action;

	return mTargetPositions;
}
Eigen::VectorXd
Character::
computeAvgVelocity(const Eigen::VectorXd& p0, const Eigen::VectorXd& p1, double dt)
{
	Eigen::VectorXd vel = Eigen::VectorXd::Zero(p0.rows());
	for(int i=0;i<mSkeleton->getNumJoints();i++)
	{
		if(mSkeleton->getJoint(i)->getType()=="BallJoint")
		{
			int idx = mSkeleton->getJoint(i)->getIndexInSkeleton(0);
			Eigen::Matrix3d R0 = BallJoint::convertToRotation(p0.segment<3>(idx));
			Eigen::Matrix3d R1 = BallJoint::convertToRotation(p1.segment<3>(idx));
			Eigen::Vector3d w = 1.0/dt*dart::math::logMap(R0.transpose()*R1);

			vel.segment<3>(idx) = w;
		}
		else if(mSkeleton->getJoint(i)->getType()=="FreeJoint")
		{
			int idx = mSkeleton->getJoint(i)->getIndexInSkeleton(0);
			Eigen::Isometry3d T0 = FreeJoint::convertToTransform(p0.segment<6>(idx));
			Eigen::Isometry3d T1 = FreeJoint::convertToTransform(p1.segment<6>(idx));
			Eigen::Vector6d w = 1.0/dt*dart::math::logMap(T0.inverse()*T1);

			vel.segment<6>(idx) = w;
		}
	}
	return vel;
}
void
Character::
actuate(const Eigen::VectorXd& target_position)
{
	Eigen::VectorXd q = mSkeleton->getPositions();
	Eigen::VectorXd dq = mSkeleton->getVelocities();
	double dt = mSkeleton->getTimeStep();

	Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();

	Eigen::VectorXd qdqdt = q + dq*dt;

	Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt,target_position));
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq);
	Eigen::VectorXd ddq = M_inv*(-mSkeleton->getCoriolisAndGravityForces()+p_diff+v_diff+mSkeleton->getConstraintForces());

	Eigen::VectorXd tau = p_diff + v_diff - dt*mKv.cwiseProduct(ddq);

	tau = dart::math::clip<Eigen::VectorXd,Eigen::VectorXd>(tau,mMinForces,mMaxForces);

	mSkeleton->setForces(tau);
}


std::vector<Eigen::Vector3d>
Character::
getState()
{
	Eigen::Isometry3d T_ref = this->getReferenceTransform();

	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	int n = mSkeleton->getNumBodyNodes();
	int n_ee = mEndEffectors.size();
	std::vector<Eigen::Vector3d> ps(n),vs(n),ws(n), ee(n_ee);
	std::vector<Eigen::MatrixXd> Rs(n);

	for(int i=0;i<n;i++)
	{
		Eigen::Isometry3d Ti = T_ref_inv*(mSkeleton->getBodyNode(i)->getTransform());

		ps[i] = Ti.translation();
		Rs[i] = Ti.linear();

		vs[i] = R_ref_inv*mSkeleton->getBodyNode(i)->getLinearVelocity();
		ws[i] = R_ref_inv*mSkeleton->getBodyNode(i)->getAngularVelocity();
	}
	for(int i=0;i<n_ee;i++)
		ee[i] = T_ref_inv*mEndEffectors[i]->getCOM();
	Eigen::Vector3d p_com = T_ref_inv*mSkeleton->getCOM();
	p_com[0]=0.0; p_com[2]=0.0;
	Eigen::Vector3d v_com = R_ref_inv*mSkeleton->getCOMLinearVelocity();

	std::vector<Eigen::Vector3d> states(5*n + n_ee + 2);

	int o = 0;
	for(int i=0;i<n;i++) states[o+i] = ps[i]; o += n;
	for(int i=0;i<n;i++) states[o+i] = Rs[i].col(0); o += n;
	for(int i=0;i<n;i++) states[o+i] = Rs[i].col(1); o += n;
	for(int i=0;i<n;i++) states[o+i] = vs[i]; o += n;
	for(int i=0;i<n;i++) states[o+i] = ws[i]; o += n;
	for(int i=0;i<n_ee;i++) states[o+i] = ee[i]; o += n_ee;

	states[o+0] = p_com;
	states[o+1] = v_com;

	return states;
}

Eigen::VectorXd
Character::
getStateAMP()
{
	Eigen::Isometry3d T_ref = this->getReferenceTransform();

	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	Eigen::VectorXd p = mSkeleton->getPositions();
	Eigen::VectorXd v = mSkeleton->getVelocities();

	int n = mSkeleton->getNumBodyNodes();
	int m = (p.rows()-6)/3;
	std::vector<Eigen::VectorXd> states;
	for(int i=0;i<mSkeleton->getNumJoints();i++)
	{
		auto joint = mSkeleton->getJoint(i);
		
		if(joint->getType()=="BallJoint")
		{
			int idx = joint->getIndexInSkeleton(0);
			Eigen::Matrix3d R = dart::dynamics::BallJoint::convertToRotation(p.segment<3>(idx));

			states.emplace_back(R.col(0));
			states.emplace_back(R.col(1));
		}
		else if(joint->getType()=="FreeJoint")
		{
			int idx = joint->getIndexInSkeleton(0);
			Eigen::Matrix3d R = dart::dynamics::BallJoint::convertToRotation(p.segment<3>(idx));
			R = R_ref_inv*R;
			
			states.emplace_back(R.col(0));
			states.emplace_back(R.col(1));
		}
	}

	for(int i=0;i<mEndEffectors.size();i++)
		states.emplace_back(T_ref_inv*mEndEffectors[i]->getCOM());

	Eigen::Vector3d p_root = T_ref_inv*mSkeleton->getBodyNode(0)->getCOM();
	p_root[0] =0.0; p_root[2] =0.0;
	Eigen::Vector3d v_root = R_ref_inv*mSkeleton->getBodyNode(0)->getLinearVelocity();
	Eigen::Vector3d w_root = R_ref_inv*mSkeleton->getBodyNode(0)->getAngularVelocity();

	states.emplace_back(p_root);
	states.emplace_back(v_root);
	states.emplace_back(w_root);

	states.emplace_back(v.tail(v.rows()-6));	

	return MathUtils::ravel(states);
}

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
	mBVHNames = bvh_names;

	for(int i=0;i<mBVHMap.size();i++)
	{
		int index = std::distance(bvh_names.begin(),std::find(bvh_names.begin(),bvh_names.end(), mBVHMap[i]));
		if(index>=bvh_names.size())
			mBVHIndices.emplace_back(-1);
		else
			mBVHIndices.emplace_back(index);
	}
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