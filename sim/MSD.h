#ifndef __MSD_H__
#define __MSD_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
class CartesianMSD
{
public:
	CartesianMSD(const Eigen::Vector3d& k, const Eigen::Vector3d& d, const Eigen::Vector3d& m, double dt);

	void reset();

	void setProjection(const Eigen::Vector3d& position){mProjection = position;}
	void applyForce(const Eigen::Vector3d& f);
	void step();

	const Eigen::Vector3d& getPosition(){return mPosition;}
	const Eigen::Vector3d& getVelocity(){return mVelocity;}

	Eigen::VectorXd saveState();
	void restoreState(const Eigen::VectorXd& state);
private:
	Eigen::Vector3d mK, mD, mInvM;
	double mdt;

	Eigen::Vector3d mPosition, mProjection, mVelocity, mForce;
};
class SphericalMSD
{
public:
	SphericalMSD(const Eigen::Vector3d& k, const Eigen::Vector3d& d, const Eigen::Vector3d& m, double dt);

	void reset();

	void setProjection(const Eigen::Matrix3d& position){mProjection = position;}
	void applyForce(const Eigen::Vector3d& f);
	void step();

	static Eigen::Vector3d log(const Eigen::Matrix3d& R);
	static Eigen::Matrix3d exp(const Eigen::Vector3d& v);

	const Eigen::Matrix3d& getPosition(){return mPosition;}
	const Eigen::Vector3d& getVelocity(){return mVelocity;}

	Eigen::VectorXd saveState();
	void restoreState(const Eigen::VectorXd& state);
private:
	Eigen::Vector3d mK, mD, mInvM;
	double mdt;

	Eigen::Matrix3d mPosition, mProjection;
	Eigen::Vector3d mVelocity, mForce;

};

class GeneralizedMSD
{
public:
	GeneralizedMSD(int njoints, const Eigen::VectorXd& k, const Eigen::VectorXd& d, const Eigen::VectorXd& m, double dt);

	void reset();

	void setProjection(const Eigen::Vector3d& position, const Eigen::MatrixXd& rotation);
	void applyForce(const Eigen::VectorXd& force);
	void step();

	CartesianMSD* getRoot(){return mRoot;}
	const std::vector<SphericalMSD*> getJoints(){return mJoints;}
	SphericalMSD* getJoint(int idx){return mJoints[idx];}

	Eigen::Vector3d getPosition();
	Eigen::MatrixXd getRotation();
	Eigen::Vector3d getLinearVelocity();
	Eigen::VectorXd getAngularVelocity();

	std::vector<Eigen::VectorXd> saveState();
	void restoreState(const std::vector<Eigen::VectorXd>& state);
private:
	int mNumJoints;
	double mdt;

	CartesianMSD* mRoot;
	std::vector<SphericalMSD*> mJoints;
};
#endif