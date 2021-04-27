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
	Environment(bool enable_lower_upper_body);

	int getDimState();
	int getDimAction();
	int getDimStateAMP();
	int getDimStateAMPLowerBody();
	int getDimStateAMPUpperBody();

	void resetGoal();
	void reset(int frame=-1);

	void updateGoal();
	void step(const Eigen::VectorXd& action);
	
	double getRewardGoal();

	const Eigen::VectorXd& getState();
	const Eigen::VectorXd& getStateGoal();
	const Eigen::VectorXd& getStateAMP();
	const Eigen::VectorXd& getStateAMPLowerBody();
	const Eigen::VectorXd& getStateAMPUpperBody();

	Eigen::MatrixXd getStateAMPExpert();
	Eigen::MatrixXd getStateAMPExpertLowerBody();
	Eigen::MatrixXd getStateAMPExpertUpperBody();

	bool inspectEndOfEpisode();

	void updateObstacle();
	void generateObstacle();

	const dart::simulation::WorldPtr& getWorld(){return mWorld;}
	dart::dynamics::SkeletonPtr getObstacle(){return mObstacle;}

	Character* getSimCharacter(){return mSimCharacter;}
	Character* getKinCharacter(){return mKinCharacter;}
	dart::dynamics::SkeletonPtr getGround(){return mGround;}
	double getTargetHeading(){return mTargetHeading;}
	double getTargetSpeed(){return mTargetSpeed;}
	bool isEnableGoal(){return mEnableGoal;}
	bool isKinematics(){return mKinematics;}
	bool isEnableLowerUpperBody(){return mEnableLowerUpperBody;}
	void setKinematics(bool kin){mKinematics = kin;}
	const dart::dynamics::SkeletonPtr& getDoor(){return mDoor;}
	double getTargetDoorAngle(){return mTargetDoorAngle;}
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
	bool mKinematics;
	Character *mSimCharacter,*mKinCharacter;
	std::vector<Motion*> mMotions;
	std::vector<Motion*> mLowerBodyMotions;
	std::vector<Motion*> mUpperBodyMotions;

	double mGroungHeight;
	dart::dynamics::SkeletonPtr mGround;
	dart::dynamics::SkeletonPtr mDoor;
	dart::constraint::BallJointConstraintPtr mDoorConstraint;

	Eigen::VectorXd mPrevPositions, mPrevVelocities, mPrevCOM;
	Eigen::VectorXd mState, mStateGoal, mStateAMP, mStateAMPLowerBody, mStateAMPUpperBody;

	bool mContactEOE;
	bool mEnableGoal;
	bool mEnableObstacle;
	bool mEnableLowerUpperBody;

	double mRewardGoal;
	bool mEnableGoalEOE;
	std::vector<double> mRewardGoals;

	double mTargetHeading, mTargetSpeed;
	double mTargetSpeedMin, mTargetSpeedMax;
	double mSharpTurnProb, mSpeedChangeProb, mMaxHeadingTurnRate;

	double mTargetDoorAngle, mPrevDoorAngle;
	int mObstacleCount;
	dart::dynamics::SkeletonPtr mObstacle;
};

#endif