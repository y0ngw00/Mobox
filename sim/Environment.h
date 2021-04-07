#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__
#include "dart/dart.hpp"
#include <tuple>
#include "Event.h"
#include "Distribution.hpp"
#include "ForceSensor.h"

class BVH;
class Motion;
class Character;
class Environment
{
public:
	Environment();
	// For Reinforcement Learning
	int getDimState0();
	int getDimAction0();
	int getDimState1();
	int getDimAction1();

	void reset(int frame=-1);
	void step(const Eigen::VectorXd& action0, const Eigen::VectorXd& action1);

	Eigen::VectorXd getState0();
	Eigen::VectorXd getState1();

	std::map<std::string,double> getReward();

	bool inspectEndOfEpisode();
	bool isSleep();

	const std::map<std::string, std::vector<double>>& getCummulatedReward(){return mRewards;}
	
	
	const dart::simulation::WorldPtr& getWorld(){return mWorld;}
	Distribution1D<int>* getInitialStateDistribution(){return mInitialStateDistribution;}
	Character* getSimCharacter(){return mSimCharacter;}
	Character* getKinCharacter(){return mKinCharacter;}
	dart::dynamics::SkeletonPtr getGround(){return mGround;}
	Event* getEvent(){return mEvent;}

	void setKinematics(bool kin){mKinematic = kin;}
	bool getKinematics(){return mKinematic;}

	const dart::dynamics::SkeletonPtr& getDoor(){return mDoor;}
private:
	// For Reinforcement Learning
	Eigen::MatrixXd getActionSpace0();
	Eigen::VectorXd getActionWeight0();
	Eigen::VectorXd convertToRealActionSpace0(const Eigen::VectorXd& a_norm);
	
	Eigen::MatrixXd getActionSpace1();
	Eigen::VectorXd getActionWeight1();
	Eigen::VectorXd convertToRealActionSpace1(const Eigen::VectorXd& a_norm);

	Eigen::MatrixXd mActionSpace0, mActionSpace1;
	Eigen::VectorXd mActionWeight0, mActionWeight1;

	dart::simulation::WorldPtr mWorld;
	int mControlHz, mSimulationHz;
	int mElapsedFrame, mStartFrame, mCurrentFrame;
	int mMaxElapsedFrame;
	Character *mSimCharacter,*mKinCharacter;

	dart::dynamics::SkeletonPtr mGround;
	dart::dynamics::SkeletonPtr mDoor;
	dart::constraint::BallJointConstraintPtr mDoorConstraint;
	bool mDoorConstraintOn;
	std::vector<int> mImitationFrameWindow;
	Distribution1D<int>* mInitialStateDistribution;

	double mWeightPos, mWeightVel, mWeightEE, mWeightRoot, mWeightCOM;

	std::map<std::string, std::vector<double>> mRewards;

	Eigen::VectorXd mState0, mState1;
	bool mState0Dirty, mState1Dirty;
	bool mKinematic;
	Event* mEvent;

	Eigen::Vector3d mPredefinedAction;
};

#endif