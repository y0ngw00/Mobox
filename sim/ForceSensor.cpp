#include "ForceSensor.h"

ForceSensor::
ForceSensor(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& local_pos)
	:mBodyNode(bn), mLocalOffset(local_pos),
	mh(bn->getSkeleton()->getTimeStep()),
	mX(Eigen::Vector3d::Zero()),
	mV(Eigen::Vector3d::Zero()),
	mfext(Eigen::Vector3d::Zero()),
	mM(0.1),mK(500.0),md(0.1),
	mInsomnia(false)
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
	mfext1.setZero();
	mSleep = true;
}
void
ForceSensor::
addExternalForce(const Eigen::Vector3d& f_ext)
{
	Eigen::Vector3d f = f_ext;
	// mfext1 = 0.1*f + 0.9*mfext1;
	// for(int i=0;i<3;i++)
	// {
	// 	bool sign = f[i]<0.0?true:false;
	// 	f[i] = std::log(std::abs(f[i]) + 1.0);
	// 	if(sign)
	// 		f[i] = -f[i];
	// }
	// std::cout<<f.norm()<<std::endl;
	// std::cout<<f.norm()<<std::endl;
	// if(f.norm()>20)
	// 	f *= (20.0/f.norm());
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

	double x_norm = mX.norm();
	double v_norm = mV.norm();
	// std::cout<<x_norm<<std::endl;
	if(x_norm>5e-1)
		mX *= 5e-1/x_norm;
	if(v_norm>1.0)
		mV *= 1.0/v_norm;
	mfext.setZero();
}