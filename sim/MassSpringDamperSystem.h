#ifndef __MASS_SPRING_DAMPER_SYSTEM_H__
#define __MASS_SPRING_DAMPER_SYSTEM_H__
#include "dart/dart.hpp"
class TwoJointIK;
class Character;

class CartesianMSDSystem
{
public:
	CartesianMSDSystem(const Eigen::Vector3d& mass_coeffs,
		const Eigen::Vector3d& spring_coeffs,
		const Eigen::Vector3d& damper_coeffs,
		double timestep);

	void reset();
	void applyForce(const Eigen::Vector3d& force);

	void step();

	Eigen::VectorXd getState(const Eigen::Isometry3d& T_ref = Eigen::Isometry3d::Identity());
	Eigen::VectorXd saveState();
	void restoreState(const Eigen::VectorXd& state);
	const Eigen::Vector3d& getPosition(){return mPosition;}
	const Eigen::Vector3d& getVelocity(){return mVelocity;}

	void setPosition(const Eigen::Vector3d& position){mPosition = position;}
	void setVelocity(const Eigen::Vector3d& velocity){mVelocity = velocity;}
public:

	double mTimestep;
	Eigen::Vector3d mMassCoeffs, mSpringCoeffs, mDamperCoeffs;

	Eigen::Vector3d mPosition, mVelocity, mForce;
};

class SphericalMSDSystem
{
public:
	SphericalMSDSystem(double mass_coeff, 
						double spring_coeff,
						double damper_coeff,double timestep);

	void reset();
	void applyForce(const Eigen::Vector3d& f);

	void step();

	Eigen::VectorXd getState();
	Eigen::VectorXd saveState();
	void restoreState(const Eigen::VectorXd& state);
	
	const Eigen::Matrix3d& getPosition(){return mPosition;}
	const Eigen::Vector3d& getVelocity(){return mVelocity;}

	void setPosition(const Eigen::Matrix3d& position){mPosition = position;}
	void setVelocity(const Eigen::Vector3d& velocity){mVelocity = velocity;}
public:
	double mTimestep;
	double mMassCoeff, mSpringCoeff, mDamperCoeff;

	Eigen::Matrix3d mPosition;
	Eigen::Vector3d mVelocity;
	Eigen::Vector3d mForce;
};




class MassSpringDamperSystem
{
public:
	MassSpringDamperSystem(Character* character,
		const Eigen::VectorXd& mass_coeffs,
		const Eigen::VectorXd& spring_coeffs,
		const Eigen::VectorXd& damper_coeffs,
		double timestep);

	void reset();

	void applyForce(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& force, const Eigen::Vector3d& offset);
	std::pair<Eigen::Vector3d, Eigen::MatrixXd> step(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR);

	const Eigen::MatrixXd& getR(){return mR;}
	const Eigen::MatrixXd& getw(){return mw;}
	const Eigen::Vector3d& getp(){return mPp;}
	const Eigen::Vector3d& getv(){return mPv;}
public:
	
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mSkeleton;

	int mNumJoints;
	Eigen::MatrixXd mR; //3*3n rotation
	Eigen::MatrixXd mw; //3*n angular velocity;
	Eigen::MatrixXd mf; //3*n generalized forces

	int mCount;
	double mTimestep;
	Eigen::VectorXd mMassCoeffs, mSpringCoeffs, mDamperCoeffs;

	Eigen::Vector3d mPp, mPv, mPf;
	double mPMassCoeffs, mPSpringCoeffs, mPDamperCoeffs;


	//For stepping strategy
	std::vector<TwoJointIK*> mTwoJointIKs;
	double mStepTime,mphase;
	int mSwing, mStance;
	Eigen::Isometry3d mSwingPosition, mStancePosition;
	Eigen::Vector3d mCurrentHipPosition,mGlobalHipPosition;
	Eigen::MatrixXd mR_IK0, mR_IK1;
	Eigen::Vector3d mP_IK0, mP_IK1;
	bool mFootChanged;
	bool mSolveFootIK;
	int mFootState;

	Eigen::Vector3d uT, xT;
	// Eigen::MatrixXd mR_IK, mR_IK1;


};
#endif