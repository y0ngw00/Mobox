#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__
#include "dart/dart.hpp"
#include <tuple>

class BVH;
class Motion;
class Character;
class CartesianMSDSystem;
class Environment
{
public:
	Environment();

	int getDimState();
	int getDimAction();
	int getDimStateAMP();

	void reset(int frame=-1);

	void step(const Eigen::VectorXd& action);
	
	void resetGoal();
	void updateGoal();
	double getRewardGoal();

	const Eigen::VectorXd& getState();
	const Eigen::VectorXd& getStateGoal();
	const Eigen::VectorXd& getStateAMP();

	Eigen::MatrixXd getStateAMPExpert();

	bool inspectEndOfEpisode();

	const dart::simulation::WorldPtr& getWorld(){return mWorld;}

	Character* getSimCharacter(){return mSimCharacter;}
	Character* getKinCharacter(){return mKinCharacter;}
	dart::dynamics::SkeletonPtr getGround(){return mGround;}

	bool isEnableGoal(){return mEnableGoal;}
private:
	double computeGroundHeight();
	void recordState();
	void recordGoal();

	Eigen::MatrixXd getActionSpace();
	Eigen::VectorXd getActionWeight();
	Eigen::VectorXd convertToRealActionSpace(const Eigen::VectorXd& a_norm);

	Eigen::MatrixXd mActionSpace;
	Eigen::VectorXd mActionWeight;

	dart::simulation::WorldPtr mWorld;
	int mControlHz, mSimulationHz;
	int mElapsedFrame, mFrame;
	int mMaxElapsedFrame;

	Character *mSimCharacter,*mKinCharacter;
	std::vector<Motion*> mMotions;

	dart::dynamics::SkeletonPtr mGround;

	Eigen::VectorXd mPrevPositions, mPrevPositions2, mPrevCOM;
	Eigen::VectorXd mState, mStateGoal, mStateAMP;

	bool mContactEOE;
	bool mEnableGoal;

	double mRewardGoal;

	double mTargetHeading, mTargetSpeed;
	double mTargetSpeedMin, mTargetSpeedMax;
	double mTargetFrame;
	double mSharpTurnProb, mSpeedChangeProb, mMaxHeadingTurnRate;
};
#endif