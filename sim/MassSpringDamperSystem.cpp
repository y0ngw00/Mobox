#include "MassSpringDamperSystem.h"
#include "Character.h"
using namespace dart::dynamics;
MassSpringDamperSystem::
MassSpringDamperSystem(Character* character,
					const Eigen::VectorXd& mass_coeffs,
					const Eigen::VectorXd& spring_coeffs,
					const Eigen::VectorXd& damper_coeffs,
					double timestep)
	:mCharacter(character), mNumJoints(mCharacter->getSkeleton()->getNumJoints()),
	mMassCoeffs(mass_coeffs*0.5),mSpringCoeffs(spring_coeffs*2.0),mDamperCoeffs(damper_coeffs),mTimestep(timestep) // FFF
{
	mR.resize(3,3*mNumJoints);	
	mw.resize(3,mNumJoints);
	mf.resize(3,mNumJoints);

	mPMassCoeffs = 100.0;
	mPSpringCoeffs = 800.0;
	mPDamperCoeffs = 200.0;
	this->reset();
}
void
MassSpringDamperSystem::
reset()
{
	for(int i=0;i<mNumJoints;i++)
		mR.block<3,3>(0,3*i) = Eigen::Matrix3d::Identity();

	mw.setZero();
	mf.setZero();

	mPp.setZero();
	mPv.setZero();
	mPf.setZero();
}

void
MassSpringDamperSystem::
applyForce(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& force, const Eigen::Vector3d& offset)
{
	dart::math::LinearJacobian J = mCharacter->getSkeleton()->getLinearJacobian(bn, offset);
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::VectorXd s = svd.singularValues();
	Eigen::Matrix3d s_inv = Eigen::Matrix3d::Zero();
	for(int i =0;i<3;i++)
	{
		if(s[0]>1e-6) s_inv(i,i) = 1.0/s[0];
		else s_inv(i,i) = 0.0;
	}
	
	Eigen::MatrixXd Jt = svd.matrixV()*s_inv*(svd.matrixU().transpose());
	Eigen::VectorXd Jtf = Jt*(force);
	for(int i=0;i<mNumJoints;i++)
	{
		auto joint = mCharacter->getSkeleton()->getJoint(i);
		
		int idx = mCharacter->getBVHIndex(i);
		if(joint->getType()!="BallJoint" || idx<0)
			continue;
		int idx_in_jac = joint->getIndexInSkeleton(0);
		mf.col(idx) += Jtf.segment<3>(idx_in_jac);
	}

	mPf = force;
	// mPf.setZero();
}

std::pair<Eigen::Vector3d, Eigen::MatrixXd>
MassSpringDamperSystem::
step(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR)
{
	int cnt = 0;
	for(int i=0;i<mNumJoints;i++)
	{
		Eigen::Matrix3d Ri = mR.block<3,3>(0,i*3);
		Eigen::Vector3d aai = dart::math::logMap(Ri);
		Eigen::Vector3d wi = mw.col(i);
		Eigen::Vector3d fi = mf.col(i);
		auto joint = mCharacter->getSkeleton()->getJoint(i);
		int idx = mCharacter->getBVHIndex(i);
		if(joint->getType()!="BallJoint" || idx<0 || i>8){}
		else {
			cnt += 3;
		}

		double h = mTimestep;
		double m = mMassCoeffs[i];
		double k = mSpringCoeffs[i];
		double d = mDamperCoeffs[i];
		double angle = aai.norm();

		wi += h/m*(fi);
		wi = dart::math::clip<Eigen::Vector3d>(wi, Eigen::Vector3d::Constant(-20.0), Eigen::Vector3d::Constant(20.0));
		wi *= 0.7;
		Ri = Ri*dart::math::expMapRot(h*wi);

		Eigen::Vector3d ri = dart::math::logMap(Ri);

		mR.block<3,3>(0,i*3) = Ri;
		mw.col(i) = wi;
	}
	mPv += mTimestep/mPMassCoeffs*(-mPSpringCoeffs*mPp + mPf);

	mPv *= 0.6;
	mPp += mTimestep*mPv;
	
	mf.setZero();
	mPf.setZero();

	Eigen::MatrixXd R(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		R.block<3,3>(0,i*3) = baseR.block<3,3>(0,i*3)*mR.block<3,3>(0,i*3);

	return std::make_pair(mPp+baseP, R);
}
