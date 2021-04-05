#ifndef __FORCE_SENSOR_H__
#define __FORCE_SENSOR_H__
#include "dart/dart.hpp"

class ForceSensor
{
public:
	ForceSensor(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& local_pos);

	void setInsomnia(bool insomnia){mInsomnia = insomnia;}
	void reset();
	void addExternalForce(const Eigen::Vector3d& f_ext);
	void step();

	bool isSleep(){if(mInsomnia)return false; return mSleep;}
	const Eigen::Vector3d& getLocalOffset(){return mLocalOffset;}
	dart::dynamics::BodyNode* getBodyNode(){return mBodyNode;}
	Eigen::Vector3d getHapticPosition(bool local = true);
	Eigen::Vector3d getHapticVelocity(bool local = true);

	Eigen::Vector3d getVelocity(){return mBodyNode->getLinearVelocity(mLocalOffset);}
	Eigen::Vector3d getPosition(){return mBodyNode->getTransform()*mLocalOffset;}
private:
	dart::dynamics::BodyNode* mBodyNode;
	Eigen::Vector3d mLocalOffset;

	// Internal haptic states
	Eigen::Vector3d mX, mV;
	Eigen::Vector3d mfext, mfext1;
	bool mSleep;
	bool mInsomnia;
	double mh;
	double mM,mK;
	double md;

	Eigen::Matrix2d mA_inv;

};


#endif