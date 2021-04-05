#ifndef __MASS_SPRING_DAMPER_SYSTEM_H__
#define __MASS_SPRING_DAMPER_SYSTEM_H__
#include "dart/dart.hpp"
class Character;
class MassSpringDamperSystem
{
public:
	MassSpringDamperSystem(Character* character,
		const Eigen::VectorXd& mass_coeffs,
		const Eigen::VectorXd& spring_coeffs,
		const Eigen::VectorXd& damper_coeffs,
		double timestep);

	void reset();

	int getNumDofs();
	void addTargetPose(const Eigen::VectorXd& pose);
	const Eigen::MatrixXd& getTargetPose(){return mTargetPose;}
	void applyForce(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& force, const Eigen::Vector3d& offset);
	// std::pair<Eigen::Vector3d, Eigen::MatrixXd> step(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR);
	std::pair<Eigen::Vector3d, Eigen::MatrixXd> stepForce(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR, const Eigen::VectorXd& weights);
	std::pair<Eigen::Vector3d, Eigen::MatrixXd> stepPose(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR);

	const Eigen::MatrixXd& getR(){return mR;}
	const Eigen::MatrixXd& getw(){return mw;}
public:
	
	Character* mCharacter;

	int mNumJoints;
	Eigen::MatrixXd mR; //3*3n rotation
	Eigen::MatrixXd mw; //3*n angular velocity;
	Eigen::MatrixXd mf; //3*n generalized forces

	double mTimestep;
	Eigen::VectorXd mMassCoeffs, mSpringCoeffs, mDamperCoeffs;
	Eigen::VectorXd mJacobianWeights;

	Eigen::Vector3d mPp, mPv, mPf;
	double mPMassCoeffs, mPSpringCoeffs, mPDamperCoeffs;

	int mNumDofs;
	Eigen::MatrixXd mTargetPose;
};
#endif