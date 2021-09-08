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
	int getDimStateAMPLabel();


	const Eigen::Vector3d getTargetDirection();
	const Eigen::VectorXd getTargetMotion();

	void setTargetMotion(const Eigen::VectorXd motion_type);
	void addStateGoal(int idx, double value);
	void setControl() {isControl = true;}


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
	std::vector<std::string> mEndEffector;

	dart::dynamics::SkeletonPtr mGround;

	Eigen::VectorXd mPrevPositions, mPrevPositions2, mPrevCOM;
	Eigen::VectorXd mState, mStateGoal, mStateAMP, mStateLabel;

	bool mContactEOE;
	bool mEnableGoal;

	double mRewardGoal;

	int mNumMotions, mNumAddInfo;

	double mTargetHeading, mTargetSpeed;
	Eigen::Vector3d mTargetDirection;
	double mTargetSpeedMin, mTargetSpeedMax;
	double mTargetFrame;
	double mTargetHeight, mIdleHeight;
	double mSharpTurnProb, mSpeedChangeProb,mHeightChangeProb, mMaxHeadingTurnRate;

	double mTargetHeightMin, mTargetHeightMax;
	bool isControl;
	int dur;
	int mControlidx;
	double mControlValue;

};
#endif