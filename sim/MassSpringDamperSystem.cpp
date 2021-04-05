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
	// mMassCoeffs(mass_coeffs*0.5),mSpringCoeffs(spring_coeffs*10.0),mDamperCoeffs(damper_coeffs),mTimestep(timestep) // PPP
{
	mR.resize(3,3*mNumJoints);	
	mw.resize(3,mNumJoints);
	mf.resize(3,mNumJoints);

	mPMassCoeffs = 100.0;
	mPSpringCoeffs = 800.0;
	mPDamperCoeffs = 200.0;
	mJacobianWeights = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
	int index = 0;
	index = mCharacter->getSkeleton()->getJoint("Spine1")->getIndexInSkeleton(0);
	mJacobianWeights.segment<3>(index) = Eigen::Vector3d::Constant(0.2);
	index = mCharacter->getSkeleton()->getJoint("Head")->getIndexInSkeleton(0);
	mJacobianWeights.segment<3>(index) = Eigen::Vector3d::Constant(0.4);
	index = mCharacter->getSkeleton()->getJoint("LeftArm")->getIndexInSkeleton(0);
	mJacobianWeights.segment<3>(index) = Eigen::Vector3d::Constant(1.0);
	index = mCharacter->getSkeleton()->getJoint("LeftForeArm")->getIndexInSkeleton(0);
	mJacobianWeights.segment<3>(index) = Eigen::Vector3d::Constant(1.0);
	index = mCharacter->getSkeleton()->getJoint("RightArm")->getIndexInSkeleton(0);
	mJacobianWeights.segment<3>(index) = Eigen::Vector3d::Constant(1.0);
	index = mCharacter->getSkeleton()->getJoint("RightForeArm")->getIndexInSkeleton(0);
	mJacobianWeights.segment<3>(index) = Eigen::Vector3d::Constant(1.0);
	mTargetPose = Eigen::MatrixXd::Zero(0, 0);
	mNumDofs = -1;
	this->reset();

}
int
MassSpringDamperSystem::
getNumDofs()
{
	if(mNumDofs>=0)
		return mNumDofs;

	int cnt = 0;
	for(int i=0;i<mNumJoints;i++)
	{
		auto joint = mCharacter->getSkeleton()->getJoint(i);
		
		int idx = mCharacter->getBVHIndex(i);
		if(joint->getType()!="BallJoint" || idx<0)
			continue;
		if(i>8) break;
		cnt++;
	}
	return cnt*3;
}
void
MassSpringDamperSystem::
addTargetPose(const Eigen::VectorXd& pose)
{
	int cnt = 0;
	// for(int i=0;i<mNumJoints;i++)
	// 	mTargetPose.block<3,3>(0,3*i) = Eigen::Matrix3d::Identity();
	for(int i=0;i<mNumJoints;i++)
	{
		auto joint = mCharacter->getSkeleton()->getJoint(i);
		
		int idx = mCharacter->getBVHIndex(i);
		if(joint->getType()!="BallJoint" || idx<0)
			continue;
		if(i>8) break;
		//Slerp 
		// Eigen::Matrix3d R1 = mTargetPose.block<3, 3>(0, idx*3);
		// Eigen::Matrix3d R2 = dart::math::expMapRot(pose.segment<3>(cnt*3));
		// Eigen::AngleAxisd aa12 = Eigen::AngleAxisd(R1.transpose()*R2);
		// aa12.angle() *= 0.1;

		// mTargetPose.block<3, 3>(0, idx*3) = R1*aa12.toRotationMatrix();
		mTargetPose.block<3, 3>(0, idx*3) = dart::math::expMapRot(pose.segment<3>(cnt*3));
		cnt++;
	}

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
	mTargetPose = Eigen::MatrixXd::Zero(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		mTargetPose.block<3,3>(0,3*i) = Eigen::Matrix3d::Identity();
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
	
	// Eigen::MatrixXd inv_mass_matrix = mCharacter->getSkeleton()->getInvMassMatrix();
	Eigen::MatrixXd Jt = svd.matrixV()*s_inv*(svd.matrixU().transpose());
	// Jt.col(0) = Jt.col(0).cwiseProduct(mJacobianWeights);
	// Jt.col(1) = Jt.col(1).cwiseProduct(mJacobianWeights);
	// Jt.col(2) = Jt.col(2).cwiseProduct(mJacobianWeights);
	Eigen::VectorXd Jtf = Jt*(force);
	// std::cout<<Jtf.transpose()<<std::endl;
	// std::cout<<force.transpose()<<std::endl;
	for(int i=0;i<mNumJoints;i++)
	{
		auto joint = mCharacter->getSkeleton()->getJoint(i);
		
		int idx = mCharacter->getBVHIndex(i);
		if(joint->getType()!="BallJoint" || idx<0)
			continue;
		int idx_in_jac = joint->getIndexInSkeleton(0);
		mf.col(idx) += Jtf.segment<3>(idx_in_jac);
	}

	// mPf = force;
	// mPf.setZero();
}

// std::pair<Eigen::Vector3d, Eigen::MatrixXd>
// MassSpringDamperSystem::
// step(const Eigen::Vector3d& baseP,const Eigen::MatrixXd& baseR)
// {
// 	if(mTargetPose.rows()!=0)
// 		return stepPose(baseP, baseR);
// 	else
// 		return stepForce(baseP, baseR);
	
// }
std::pair<Eigen::Vector3d, Eigen::MatrixXd>
MassSpringDamperSystem::
stepPose(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR)
{

	for(int i=0;i<mNumJoints;i++)
	{
		if(i>8) break;
		Eigen::Matrix3d Ri = mR.block<3,3>(0, i*3);
		Eigen::Matrix3d Rt = mTargetPose.block<3,3>(0, i*3);
		Eigen::Vector3d aai = dart::math::logMap((Rt.transpose())*Ri);
		Eigen::Vector3d wi = mw.col(i);
		// std::cout<<i<<"\n"<<Rt<<std::endl<<std::endl;
		double h = mTimestep;
		double m = mMassCoeffs[i];
		double k = mSpringCoeffs[i];
		double d = mDamperCoeffs[i];
		double angle = aai.norm();

		wi += h/m*(-k*aai);
		// std::cout<<i<<"\n"<<wi.transpose()<<std::endl<<std::endl;

		// wi = dart::math::clip<Eigen::Vector3d>(wi, Eigen::Vector3d::Constant(-20.0), Eigen::Vector3d::Constant(20.0));
		wi *= 0.9;
		Ri = Ri*dart::math::expMapRot(h*wi);

		mR.block<3,3>(0,i*3) = Ri;
		mw.col(i) = wi;
	}
	Eigen::MatrixXd R(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		R.block<3,3>(0,i*3) = baseR.block<3,3>(0,i*3)*mR.block<3,3>(0,i*3);
	for(int i=0;i<mNumJoints;i++)
		mTargetPose.block<3,3>(0,3*i) = Eigen::Matrix3d::Identity();
	return std::make_pair(baseP, R);
}
std::pair<Eigen::Vector3d, Eigen::MatrixXd>
MassSpringDamperSystem::
stepForce(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR, const Eigen::VectorXd& weights)
{
	int cnt = 0;
	for(int i=0;i<mNumJoints;i++)
	{
		Eigen::Matrix3d Ri = mR.block<3,3>(0,i*3);
		Eigen::Vector3d aai = dart::math::logMap(Ri);
		Eigen::Vector3d wi = mw.col(i);
		Eigen::Vector3d fi = mf.col(i);
		Eigen::Vector3d weighti = Eigen::Vector3d::Ones();
		auto joint = mCharacter->getSkeleton()->getJoint(i);
		int idx = mCharacter->getBVHIndex(i);
		if(joint->getType()!="BallJoint" || idx<0 || i>8){}
		else {
			weighti = weights.segment<3>(cnt);
			cnt += 3;
		}

		double h = mTimestep;
		double m = mMassCoeffs[i];
		double k = mSpringCoeffs[i];
		double d = mDamperCoeffs[i];
		double angle = aai.norm();

		// Implicit Time integration
		// Eigen::Matrix3d A;
		// A = (1.0 - h*d/m - h*k/m)*Eigen::Matrix3d::Identity();

		// A = (1.0 + h*d/m)*Eigen::Matrix3d::Identity() + h*k/m*Ri;
		// wi = A.ldlt().solve(wi - h*k/m*aai + h/m*fi);

		// wi += h/m*(-k*aai + fi);
		wi += h/m*(fi);
		wi = dart::math::clip<Eigen::Vector3d>(wi, Eigen::Vector3d::Constant(-20.0), Eigen::Vector3d::Constant(20.0));
		wi *= 0.7;
		Ri = Ri*dart::math::expMapRot(h*wi);

		Eigen::Vector3d ri = dart::math::logMap(Ri);
		// double r_norm = ri.norm();
		// if(r_norm>M_PI*0.8){
		// 	ri *= M_PI*0.8/r_norm;
		// 	Ri = dart::math::expMapRot(ri);
		// }
		mR.block<3,3>(0,i*3) = Ri;
		mw.col(i) = wi;
	}
	mPv += mTimestep/mPMassCoeffs*(-mPSpringCoeffs*mPp + mPf);


	Eigen::Vector3d center(0.0,-0.4,0.0);
	Eigen::Vector3d radius(0.2,0.4,0.2);

	Eigen::Vector3d lb, ub;
	lb[0] = -0.4;
	lb[1] = -0.6;
	lb[2] = -0.4;

	ub[0] = 0.4;
	ub[1] = 0.0;
	ub[2] = 0.4;

	for(int i=0;i<3;i++)
	{
		if(mPp[i]>ub[i])
			mPv[i] += -0.4*(mPp[i]-ub[i])/mTimestep;
		else if(mPp[i]<lb[i])
			mPv[i] += -0.4*(mPp[i]-lb[i])/mTimestep;		
	}
	mPv *= 0.9;
	mPp += mTimestep*mPv;
	
	mf.setZero();
	mPf.setZero();

	Eigen::MatrixXd R(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		R.block<3,3>(0,i*3) = baseR.block<3,3>(0,i*3)*mR.block<3,3>(0,i*3);

	return std::make_pair(mPp+baseP, R);
}
/*void
MassSpringDamperSystem::
step()
{
	for(int i=0;i<mNumJoints;i++)
	{
		Eigen::Matrix3d Ri = mR.block<3,3>(0,i*3);
		Eigen::Vector3d aai = dart::math::logMap(Ri);
		Eigen::Vector3d wi = mw.col(i);
		Eigen::Vector3d fi = mf.col(i);
		
		wi += mTimestep/mMassCoeffs[i]*(-mSpringCoeffs[i]*aai - mDamperCoeffs[i]*wi + fi);
		Ri = Ri*dart::math::expMapRot(h*wi);

		mR.block<3,3>(0,i*3) = Ri;
		mw.col(i) = wi;
	}
	mf.setZero();
}*/