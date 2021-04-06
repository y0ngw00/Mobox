#ifndef __CHARACTER_H__
#define __CHARACTER_H__
#include "dart/dart.hpp"
#include "ForceSensor.h"
class MassSpringDamperSystem;
class Motion;
class Character
{
public:

	Character(dart::dynamics::SkeletonPtr& skel,
			const std::vector<dart::dynamics::BodyNode*>& end_effectors,
			const std::vector<std::string>& bvh_map,
			const Eigen::VectorXd& w_joint,
			const Eigen::VectorXd& kp,
			const Eigen::VectorXd& maxf);

	void addForceSensor(const Eigen::Vector3d& point);
	void setMSDParameters(const Eigen::VectorXd& mass_coeffs,
		const Eigen::VectorXd& spring_coeffs,
		const Eigen::VectorXd& damper_coeffs);
	void setBaseMotionAndCreateMSDSystem(Motion* m);
	MassSpringDamperSystem* getMSDSystem();
	// Eigen::VectorXd XXXX;
	Eigen::Vector3d XXXX;
	Motion* getMotion(){return mMotion;}
	void stepMotion(const Eigen::VectorXd& action);
	void resetMotion(int start_frame);

	const Eigen::Vector3d& getPosition(int idx);
	const Eigen::MatrixXd& getRotation(int idx);
	const Eigen::Vector3d& getLinearVelocity(int idx);
	const Eigen::MatrixXd& getAngularVelocity(int idx);

	Eigen::Isometry3d getRootTransform();
	void setRootTransform(const Eigen::Isometry3d& T);

	Eigen::Isometry3d getReferenceTransform();
	void setReferenceTransform(const Eigen::Isometry3d& T_ref);

	void setPose(const Eigen::Vector3d& position,
				const Eigen::MatrixXd& rotation);
	void setPose(const Eigen::Vector3d& position,
				const Eigen::MatrixXd& rotation,
				const Eigen::Vector3d& linear_velocity,
				const Eigen::MatrixXd& angular_velocity);

	std::pair<Eigen::VectorXd,Eigen::VectorXd> computeTargetPosAndVel(const Eigen::MatrixXd& base_rot, const Eigen::VectorXd& action,
									const Eigen::MatrixXd& angular_velocity);
	void actuate(const Eigen::VectorXd& target_position,
				const Eigen::VectorXd& target_velocity);

	std::map<std::string, Eigen::MatrixXd> getStateBody();
	std::map<std::string, Eigen::MatrixXd> getStateJoint();
	std::map<std::string, Eigen::MatrixXd> getStateForceSensors();

	std::vector<Eigen::Vector3d> getState();
	// std::vector<Eigen::Vector3d> getStateForceSensor();

	Eigen::VectorXd saveState();
	void restoreState(const Eigen::VectorXd& state);

	void buildBVHIndices(const std::vector<std::string>& bvh_names);

	const Eigen::VectorXd& getJointWeights(){return mJointWeights;}
	dart::dynamics::SkeletonPtr getSkeleton(){return mSkeleton;}

	const Eigen::VectorXd& getTargetPositions(){return mTargetPositions;}
	const Eigen::VectorXd& getAppliedForces(){return mAppliedForces;}
	const std::vector<dart::dynamics::BodyNode*>& getEndEffectors(){return mEndEffectors;}
	int getBVHIndex(int idx){return mBVHIndices[idx];}
	const std::vector<int>& getBVHIndices(){return mBVHIndices;}

	const std::vector<ForceSensor*>& getForceSensors(){return mForceSensors;}
	ForceSensor* getClosestForceSensor(const Eigen::Vector3d& point); // for point force
	std::pair<ForceSensor*,double> getClosestForceSensor(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1); // for ray
private:
	Eigen::VectorXd toSimPose(const Eigen::Vector3d& position, const Eigen::MatrixXd& rotation);
	Eigen::VectorXd toSimVel(const Eigen::Vector3d& position, const Eigen::MatrixXd& rotation, const Eigen::Vector3d& linear_velocity, const Eigen::MatrixXd& angular_velcity);

	dart::dynamics::SkeletonPtr mSkeleton;

	std::vector<dart::dynamics::BodyNode*> mEndEffectors;
	std::vector<std::string> mBVHMap;
	std::vector<int> mBVHIndices;


	Eigen::VectorXd mJointWeights;

	Eigen::VectorXd mKp, mKv, mMinForces, mMaxForces, mAppliedForces, mTargetPositions;

	std::vector<ForceSensor*> mForceSensors;

	Eigen::VectorXd mMassParams,mDamperParams,mSpringParams;
	MassSpringDamperSystem* mMSDSystem;
	Motion* mMotion, *mMSDMotion;
	int mMotionCounter, mMotionStartFrame;
	int mResponseDelay;
};

#endif