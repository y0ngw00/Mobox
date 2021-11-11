#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__
#include "dart/dart.hpp"
#include <tuple>
#include <map>

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
	int getDimStateLabel();
	int getNumTotalLabel();

	double getTargetHeading();
	double getTargetHeight();
	double getTargetSpeed();
	std::vector<std::string> getMotionLabels();
	const Eigen::Vector3d getTargetDirection();

	void setTargetHeading(double heading);
	void setTargetSpeed(double speed);
	void setTargetHeight(double height);


	void reset(bool RSI = true);
	void FollowBVH(int idx);
	void readLabelFile(std::string txt_path);
	void ParseLabel(std::string filename,std::vector<Eigen::VectorXd>& label_info);

	void step(const Eigen::VectorXd& action);
	
	void resetGoal();
	void updateGoal();
	double getRewardGoal();
	int getStateLabel();
	void setStateLabel(int label);

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

	int mNumMotions, mStateLabel, mNumFeatureAdds,mDimLabel;

	double mTargetHeading;
	Eigen::Vector3d mTargetDirection;
	double mTargetFrame;
	double mTargetHeight, mIdleHeight;
	double mSharpTurnProb, mSpeedChangeProb,mHeightChangeProb, mMaxHeadingTurnRate, mTransitionProb;

	double mTargetHeightMin, mTargetHeightMax;

	std::vector<std::string> labels;
	std::vector<std::string> strike_bodies;
	std::vector<std::map<std::string, std::string>> mLabelMap;
	std::vector<Eigen::VectorXd> label_info;

	Eigen::Matrix3d R_init;
	bool mTargetHit;
	Eigen::Vector3d mTargetPos;
	double mTargetRadius;
	double mTargetDist;
	double mTargetSpeed;
	double mTargetDistMin;
	double mTargetDistMax;


};
#endif