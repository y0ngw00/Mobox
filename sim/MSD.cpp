#include "MSD.h"
CartesianMSD::
CartesianMSD(const Eigen::Vector3d& k, const Eigen::Vector3d& d, const Eigen::Vector3d& m, double dt)
	:mK(k), mD(d), mdt(dt)
{
	for(int i=0;i<3;i++)
	{
		if(m[i]>1e6 || m[i]<1e-6)
			mInvM[i] = 0.0;
		else
			mInvM[i] = 1.0/m[i];
	}
}

void
CartesianMSD::
reset()
{
	mPosition.setZero();
	mProjection.setZero();
	mVelocity.setZero();
	mForce.setZero();
}

void
CartesianMSD::
applyForce(const Eigen::Vector3d& f)
{
	mForce += f;
}
void
CartesianMSD::
step()
{
	mVelocity += mdt*mInvM.cwiseProduct(-mK.cwiseProduct(mPosition - mProjection) + mForce);
	mVelocity = mD.cwiseProduct(mVelocity);

	mPosition += mdt*mVelocity;

	mForce.setZero();
	mProjection.setZero();
}
Eigen::VectorXd
CartesianMSD::
saveState()
{
	Eigen::VectorXd state(6);
	state<<mPosition, mVelocity;

	return state;
}
void
CartesianMSD::
restoreState(const Eigen::VectorXd& state)
{
	mPosition = state.head<3>();
	mVelocity = state.tail<3>();
}
SphericalMSD::
SphericalMSD(const Eigen::Vector3d& k, const Eigen::Vector3d& d, const Eigen::Vector3d& m, double dt)
	:mK(k), mD(d), mdt(dt)
{
	for(int i=0;i<3;i++)
	{
		if(m[i]>1e6 || m[i]<1e-6)
			mInvM[i] = 0.0;
		else
			mInvM[i] = 1.0/m[i];
	}
}
void
SphericalMSD::
reset()
{
	mPosition.setIdentity();
	mProjection.setIdentity();
	mVelocity.setZero();
	mForce.setZero();
}

void
SphericalMSD::
applyForce(const Eigen::Vector3d& f)
{
	mForce += f;
}

void
SphericalMSD::
step()
{
	mVelocity += mdt*mInvM.cwiseProduct(-mK.cwiseProduct(SphericalMSD::log(mProjection.transpose()*mPosition)) + mForce);
	mVelocity = mD.cwiseProduct(mVelocity);
	mPosition = mPosition*SphericalMSD::exp(mdt*mVelocity);
	mForce.setZero();
	mProjection.setIdentity();
}

Eigen::VectorXd
SphericalMSD::
saveState()
{
	Eigen::VectorXd state(12);
	state<<mPosition.col(0), mPosition.col(1), mPosition.col(2), mVelocity;

	return state;
}

void
SphericalMSD::
restoreState(const Eigen::VectorXd& state)
{
	mPosition.col(0) = state.segment<3>(0);
	mPosition.col(1) = state.segment<3>(3);
	mPosition.col(2) = state.segment<3>(6);
	mVelocity = state.segment<3>(9);
}

Eigen::Vector3d
SphericalMSD::
log(const Eigen::Matrix3d& R)
{
	Eigen::AngleAxisd aa(R);
	return aa.axis()*aa.angle();
}

Eigen::Matrix3d
SphericalMSD::
exp(const Eigen::Vector3d& v)
{
	if(v.norm()<1e-6)
		return Eigen::Matrix3d::Identity();
	Eigen::AngleAxisd aa(v.norm(), v.normalized());
	return aa.toRotationMatrix();
}

GeneralizedMSD::
GeneralizedMSD(int njoints, const Eigen::VectorXd& k, const Eigen::VectorXd& d, const Eigen::VectorXd& m, double dt)
	:mNumJoints(njoints), mdt(dt)
{
	mRoot = new CartesianMSD(k.head<3>(), d.head<3>(), m.head<3>(), mdt);
	for(int i=0;i<mNumJoints;i++)
		mJoints.emplace_back(new SphericalMSD(k.segment<3>(i*3+3), d.segment<3>(i*3+3), m.segment<3>(i*3+3), mdt));
}

void
GeneralizedMSD::
reset()
{
	mRoot->reset();
	for(int i=0;i<mNumJoints;i++)
		mJoints[i]->reset();	
}
void
GeneralizedMSD::
setProjection(const Eigen::Vector3d& position, const Eigen::MatrixXd& rotation)
{
	mRoot->setProjection(position);
	for(int i=0;i<mNumJoints;i++)
		mJoints[i]->setProjection(rotation.block<3,3>(0,i*3));
}
void
GeneralizedMSD::
applyForce(const Eigen::VectorXd& force)
{
	mRoot->applyForce(force.head<3>());
	for(int i=0;i<mNumJoints;i++)
		mJoints[i]->applyForce(force.segment<3>(i*3+3));
}
void
GeneralizedMSD::
step()
{
	mRoot->step();
	for(int i=0;i<mNumJoints;i++)
		mJoints[i]->step();
}
Eigen::Vector3d
GeneralizedMSD::
getPosition()
{
	return mRoot->getPosition();
}

Eigen::MatrixXd
GeneralizedMSD::
getRotation()
{
	Eigen::MatrixXd rotation(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		rotation.block<3,3>(0,i*3) = mJoints[i]->getPosition();

	return rotation;
}
Eigen::Vector3d
GeneralizedMSD::
getLinearVelocity()
{
	return mRoot->getVelocity();
}
Eigen::VectorXd
GeneralizedMSD::
getAngularVelocity()
{
	Eigen::VectorXd angvel(3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		angvel.segment<3>(i*3) = mJoints[i]->getVelocity();

	return angvel;
}

std::vector<Eigen::VectorXd>
GeneralizedMSD::
saveState()
{
	std::vector<Eigen::VectorXd> states;
	states.reserve(1 + mJoints.size());
	states.emplace_back(mRoot->saveState());
	for(int i=0;i<mJoints.size();i++)
		states.emplace_back(mJoints[i]->saveState());
	return states;
}

void
GeneralizedMSD::
restoreState(const std::vector<Eigen::VectorXd>& states)
{
	mRoot->restoreState(states[0]);
	for(int i=0;i<mJoints.size();i++)
		mJoints[i]->restoreState(states[i+1]);
}
