#include "ForceSensor.h"


ForceSensor::
ForceSensor(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& local_pos)
	:mBodyNode(bn), mLocalOffset(local_pos),
	mh(bn->getSkeleton()->getTimeStep()),
	mX(Eigen::Vector3d::Zero()),
	mV(Eigen::Vector3d::Zero()),
	mfext(Eigen::Vector3d::Zero()),
	mM(0.1),mK(500.0),md(0.1)
{
	Eigen::Matrix2d A;
	A<< 1.0, -mh,
		mh*mK/mM,1.0;
	mA_inv = A.inverse();
}

void
ForceSensor::
reset()
{
	mX.setZero();
	mV.setZero();
	mfext.setZero();
}
void
ForceSensor::
addExternalForce(const Eigen::Vector3d& f_ext)
{
	Eigen::Vector3d f = f_ext;
	mfext += mBodyNode->getTransform().linear().transpose()*f;
	mSleep = false;
}
Eigen::Vector3d
ForceSensor::
getHapticPosition(bool local)
{
	if(local)
		return mX;
	return mBodyNode->getTransform().linear()*mX;
}
Eigen::Vector3d
ForceSensor::
getHapticVelocity(bool local)
{
	if(local)
		return mV;
	return mBodyNode->getTransform().linear()*mV;
}
void
ForceSensor::
step()
{
	if(mSleep)
		return;
	for(int i=0;i<3;i++)
	{
		Eigen::Vector2d b;
		b<<mX[i], mV[i]+mh*mfext[i]/mM;
		Eigen::Vector2d s = mA_inv*b;
		mX[i] = s[0];
		mV[i] = s[1]*md;
	}
	if(mX.norm()<1e-4 && mV.norm()<1e-2)
		mSleep = true;

	mfext.setZero();
}