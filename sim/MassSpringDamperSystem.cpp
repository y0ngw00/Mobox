#include "MassSpringDamperSystem.h"
#include "Motion.h"
#include "BVH.h"
#include "Character.h"
#include "TwoJointIK.h"
using namespace dart::dynamics;

CartesianMSDSystem::
CartesianMSDSystem(const Eigen::Vector3d& mass_coeffs,
	const Eigen::Vector3d& spring_coeffs,
	const Eigen::Vector3d& damper_coeffs,
	double timestep)
	:mMassCoeffs(mass_coeffs),
	mSpringCoeffs(spring_coeffs),
	mDamperCoeffs(damper_coeffs),
	mTimestep(timestep)
{
	this->reset();
}

void
CartesianMSDSystem::
reset()
{
	mPosition.setZero();
	mVelocity.setZero();
	mForce.setZero();
}
void
CartesianMSDSystem::
applyForce(const Eigen::Vector3d& force)
{
	mForce += force;
}
void
CartesianMSDSystem::
step()
{
	for(int i=0;i<3;i++)
	{
		mVelocity[i] += mTimestep/(mMassCoeffs[i])*(-mSpringCoeffs[i]*mPosition[i] + mForce[i]);
		mVelocity[i] *= mDamperCoeffs[i];
		mPosition[i] += mTimestep*mVelocity[i];
	}
	mForce.setZero();
}
Eigen::VectorXd
CartesianMSDSystem::
saveState()
{
	Eigen::VectorXd state(6);
	state<<mPosition, mVelocity;

	return state;
}
void
CartesianMSDSystem::
restoreState(const Eigen::VectorXd& state)
{
	mPosition = state.head<3>();
	mVelocity = state.tail<3>();
}
Eigen::VectorXd
CartesianMSDSystem::
getState(const Eigen::Isometry3d& T_ref)
{
	Eigen::VectorXd state(6);
	state<<T_ref.linear().transpose()*mPosition, T_ref.linear().transpose()*mVelocity;
	return state;
}

SphericalMSDSystem::
SphericalMSDSystem(double mass_coeff, 
					double spring_coeff,
					double damper_coeff,double timestep)
	:mMassCoeff(mass_coeff),
	mSpringCoeff(spring_coeff),
	mDamperCoeff(damper_coeff),
	mTimestep(timestep)
{
	this->reset();
}


void
SphericalMSDSystem::
reset()
{
	mPosition.setIdentity();
	mVelocity.setZero();
	mForce.setZero();
}
void
SphericalMSDSystem::
applyForce(const Eigen::Vector3d& f)
{
	mForce += f;
}
void
SphericalMSDSystem::
step()
{
	mVelocity += mTimestep/(mMassCoeff)*(-mSpringCoeff*dart::math::logMap(mPosition) + mForce); //maybe incorrect?

	mVelocity *= mDamperCoeff;
	mPosition = mPosition*dart::math::expMapRot(mTimestep*mVelocity);
	mForce.setZero();
}

Eigen::VectorXd
SphericalMSDSystem::
getState()
{
	Eigen::VectorXd state(9);
	state<<mPosition.col(0), mPosition.col(1), mVelocity;

	return state;	
}
Eigen::VectorXd
SphericalMSDSystem::
saveState()
{
	Eigen::VectorXd state(12);
	state<<mPosition.col(0), mPosition.col(1), mPosition.col(2), mVelocity;

	return state;
}
void
SphericalMSDSystem::
restoreState(const Eigen::VectorXd& state)
{
	mPosition.col(0) = state.segment<3>(0);
	mPosition.col(1) = state.segment<3>(3);
	mPosition.col(2) = state.segment<3>(6);

	mVelocity = state.tail<3>();
}



























































































MassSpringDamperSystem::
MassSpringDamperSystem(Character* character,
					const Eigen::VectorXd& mass_coeffs,
					const Eigen::VectorXd& spring_coeffs,
					const Eigen::VectorXd& damper_coeffs,
					double timestep)
	:mCharacter(character),mSkeleton(character->getSkeleton()), mNumJoints(character->getSkeleton()->getNumJoints()),mStepTime(0.5),mSolveFootIK(false),
	mMassCoeffs(mass_coeffs*0.5),mSpringCoeffs(spring_coeffs*2.0),mDamperCoeffs(damper_coeffs),mTimestep(timestep) // FFF
{
	mR.resize(3,3*mNumJoints);	
	mw.resize(3,mNumJoints);
	mf.resize(3,mNumJoints);

	mPMassCoeffs = 10.0;
	mPSpringCoeffs = 0.0;
	mPDamperCoeffs = 200.0;
	this->reset();

	mTwoJointIKs.emplace_back(new TwoJointIK(character->getMotion()->getBVH(), character->getMotion()->getBVH()->getNodeIndex("simLeftFoot")));
	mTwoJointIKs.emplace_back(new TwoJointIK(character->getMotion()->getBVH(), character->getMotion()->getBVH()->getNodeIndex("simRightFoot")));
}
void
MassSpringDamperSystem::
reset()
{
	for(int i=0;i<mNumJoints;i++)
		mR.block<3,3>(0,3*i) = Eigen::Matrix3d::Identity();
	mphase = 10.0;
	mSwingPosition.setIdentity();
	mStancePosition.setIdentity();
	mSwing = 0;
	mStance = 1;
	mCount = 0;

	mCurrentHipPosition.setZero();
	mFootChanged = false;
	mFootState = 0;
	xT.setZero();
	uT.setZero();
	mR_IK0 = Eigen::MatrixXd::Zero(0,0);
	mR_IK1 = Eigen::MatrixXd::Zero(0,0);
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
	dart::math::LinearJacobian J = mSkeleton->getLinearJacobian(bn, offset);
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
		auto joint = mSkeleton->getJoint(i);
		
		int idx = mCharacter->getBVHIndex(i);
		if(joint->getType()!="BallJoint" || idx<0)
			continue;
		int idx_in_jac = joint->getIndexInSkeleton(0);
		mf.col(idx) += Jtf.segment<3>(idx_in_jac);
	}

	Eigen::Isometry3d T_ref = mCharacter->getReferenceTransform();
	Eigen::Matrix3d R_ref = T_ref.linear();

	mPf = R_ref.transpose()*force;
	// mPf.setZero();
}

std::pair<Eigen::Vector3d, Eigen::MatrixXd>
MassSpringDamperSystem::
step(const Eigen::Vector3d& baseP, const Eigen::MatrixXd& baseR)
{
	auto bvh = mCharacter->getMotion()->getBVH();
	if(mCount ==0)
	{
		int swing_foot = mSwing==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");	
		int stance_foot = mSwing==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");
		
		Eigen::Isometry3d T_swing = bvh->forwardKinematics(baseP, baseR, swing_foot)[0];
		Eigen::Isometry3d T_stance = bvh->forwardKinematics(baseP, baseR, stance_foot)[0];
		mSwingPosition = T_swing;
		mStancePosition = T_stance;
		mCount++;

		// mStepTime = std::sqrt(9.81/0.8);
	}
	int cnt = 0;
	for(int i=0;i<mNumJoints;i++)
	{
		Eigen::Matrix3d Ri = mR.block<3,3>(0,i*3);
		Eigen::Vector3d aai = dart::math::logMap(Ri);
		Eigen::Vector3d wi = mw.col(i);
		Eigen::Vector3d fi = mf.col(i);
		auto joint = mSkeleton->getJoint(i);
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
		wi += h/m*(-k*aai+fi);
		wi *= 0.7;
		Ri = Ri*dart::math::expMapRot(h*wi);

		Eigen::Vector3d ri = dart::math::logMap(Ri);

		mR.block<3,3>(0,i*3) = Ri;
		mw.col(i) = wi;
	}

	
	
	mPv += mTimestep/mPMassCoeffs*(-mPSpringCoeffs*mPp + mPf);

	mPv *= 0.6;
	mPp[0] = std::max(-0.2,std::min(0.2,mPp[0]-xT[0])) + xT[0];
	mPp[2] = std::max(-0.2,std::min(0.2,mPp[2]-xT[2])) + xT[2];
	mPp += mTimestep*mPv;
	
	mf.setZero();
	mPf.setZero();

	Eigen::MatrixXd R(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		R.block<3,3>(0,i*3) = baseR.block<3,3>(0,i*3)*mR.block<3,3>(0,i*3);


	// Adjust root position and lower body
	Eigen::Isometry3d T_ref = mCharacter->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref = T_ref.linear();

	mPp[1] = 0.0;
	Eigen::Vector3d p_xz = mPp;
	p_xz[1] = 0.0;
	mGlobalHipPosition = baseP + R_ref*p_xz;
	// std::cout<<mPp.transpose()<<std::endl;
	// mPp[1] = -0.3*p_xz.norm();

	// Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;
	if(mphase > 1.0 && (p_xz - xT).norm()>0.05)
	// if(mphase > 1.0)
	{
		if(mR_IK1.rows()==0)
		{
			if(p_xz[0]>0)
			{
				mSwing = 0;
				mStance = 1;
				mFootState = 0;
			}
			else
			{
				mSwing = 1;
				mStance = 0;
				mFootState = 1;
			}
		}
		else if(mFootState == 0)
		{
			mSwing = 1;
			mStance = 0;
			mFootState = 1;
		}
		else
		{
			mSwing = 0;
			mStance = 1;
			mFootState = 0;
		}

		mPp, mPv;
		Eigen::Vector3d x = mPp;
		Eigen::Vector3d x_dot = mPv;
		x[1] = 0.0;
		x_dot[1] = 0.0;
		int slice = 10;
		double h = mStepTime;

		double w2 = 9.81/0.8;
		Eigen::Vector3d u = Eigen::Vector3d::Zero();
		if(mR_IK1.rows()!=0)
			u = uT;

		// Eigen::MatrixXd A = Eigen::MatrixXd::Identity(6,6);
		// A.topRightCorner(3,3)    = -h*Eigen::MatrixXd::Identity(3,3);
		// A.bottomLeftCorner(3,3)  = -w2*h*Eigen::MatrixXd::Identity(3,3);
		// Eigen::MatrixXd A_inv = A.inverse();
		// Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
		// b.head<3>() = x;
		// b.tail<3>() = x_dot - w2*h*u;

		// Eigen::VectorXd s;
		// for(int i=0;i<slice;i++)
		// {
		// 	s = A_inv*b;
		// 	b.head<3>() = s.head<3>();
		// 	b.tail<3>() = s.tail<3>() - w2*h*u;
		// 	std::cout<<s.transpose()<<std::endl;
		// }
		int swing_foot = mSwing==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");	
		int stance_foot = mSwing==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");
		double n = mStepTime*300.0;
		xT = x;
		// Eigen::Vector3d x_dotT = x_dot;
		// for(int i =0;i<10;i++)
		// {
		// 	xT = xT+1.0/300.0*x_dotT;
		// 	x_dotT *= 0.8;
		// }

		Eigen::Vector3d root_p = baseP + R_ref*xT;
		Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
		Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];

		Eigen::Vector3d u_sim;
		if(mStance == 0)
			u_sim = R_ref.transpose()*(mCharacter->getSkeleton()->getBodyNode("RightFoot")->getCOM() - T_swing.translation());
		else
			u_sim = R_ref.transpose()*(mCharacter->getSkeleton()->getBodyNode("LeftFoot")->getCOM() - T_swing.translation());
		u_sim[1] = 0.0;
		u_sim = u;
		double alpha = 1.0;
		
		
		// std::cout<<uT.transpose()<<std::endl;
		Eigen::Vector3d lo, up;
		lo = Eigen::Vector3d::Constant(-0.2);
		up = Eigen::Vector3d::Constant(0.2);
		Eigen::Vector3d xtu = (xT - u_sim);
		xtu = xtu.cwiseMax(lo).cwiseMin(up);

		uT = alpha*xtu;

		if(mR_IK1.rows()==0)
		{
			mR_IK0 = R;
			mP_IK0.setZero();

			mSwingPosition = T_swing;
			mStancePosition = T_stance;

			mStancePosition.translation() -= R_ref*xT;
			mSwingPosition.translation() += R_ref*uT;
		}
		else
		{
			mR_IK0 = mR_IK1;
			mP_IK0 = mP_IK1;
		
			mStancePosition = mSwingPosition;
			// mStancePosition = T_stance;

			// mStancePosition.translation() += -R_ref*(xT-u_sim);
			// mStancePosition.translation() += -R_ref*(xT-u_sim);
			mSwingPosition = T_swing;
			mSwingPosition.translation() += R_ref*uT;
		}
		mR_IK1 = R;
		mP_IK1 = xT;


		uT = xT + uT;

		// mStepTime




		// mCurrentHipPosition = p_xz;


		// int swing_foot = mSwing==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");	
		// int stance_foot = mSwing==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");

		// Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;
		// Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
		// Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];

		// if(mR_IK1.rows()==0)
		// {
		// 	mR_IK0 = R;
		// 	mP_IK0.setZero();

		// 	mSwingPosition = T_swing;
		// 	mStancePosition = T_stance;

		// 	mSwingPosition.translation() += R_ref*mCurrentHipPosition;
		// 	mStancePosition.translation() += -R_ref*mCurrentHipPosition;
		// }
		// else
		// {
		// 	mR_IK0 = mR_IK1;
		// 	mP_IK0 = mP_IK1;

		// 	mStancePosition = mSwingPosition;
		// 	Eigen::Vector3d dir = root_p - mStancePosition.translation();
		// 	dir[1] = 0.0;
		// 	double alpha = mPv.norm();
		// 	alpha = std::max(1.0,std::min(2.0, alpha));

		// 	mSwingPosition.translation() = 2.0*dir + mStancePosition.translation();
		// 	// mSwingPosition.translation() += R_ref*mCurrentHipPosition;
		// }
		// mR_IK1 = R;
		// mP_IK1 = mCurrentHipPosition;

		mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, mR_IK1);
		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, mR_IK1);
		mphase = 0.0;
	}
		// if(mFootState == 0) //IDLE
		// {
		// 	mFootState = 1;

		// 	mCurrentHipPosition = p_xz;
		// 	if(p_xz[0]>0)
		// 	{
		// 		mSwing = 0;
		// 		mStance = 1;
		// 	}
		// 	else
		// 	{
		// 		mSwing = 1;
		// 		mStance = 0;
		// 	}

	// 		int swing_foot = mSwing==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");	
	// 		int stance_foot = mSwing==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");

	// 		Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;

	// 		Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
	// 		Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];

			

	// 		if(mR_IK1.rows()!=0)
	// 		{
	// 			mSwingPosition = T_swing;
	// 			mStancePosition = T_stance;
	// 		}
	// 		else
	// 		{

	// 		}

			

	// 		// mSwingPosition.translation() += R_ref*mCurrentHipPosition*1.3;
	// 		// mStancePosition.translation() += -R_ref*mCurrentHipPosition;

	// 		mR_IK0 = R;
	// 		mP_IK0.setZero();

	// 		mR_IK1 = R;
	// 		mP_IK1 = mCurrentHipPosition;

	// 		mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, mR_IK1);
	// 		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, mR_IK1);
	// 		mphase = 0.0;
	// 	}
	// 	else if(mFootState == 1) //ONE-STEP STEPPING
	// 	{
	// 		mFootState = 0;

	// 		mCurrentHipPosition = p_xz;
	// 		mSwing = mStance;
	// 		mStance = 1 - mStance;

	// 		int swing_foot = mSwing==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");	
	// 		int stance_foot = mSwing==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");

	// 		Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;
	// 		Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
	// 		Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];

	// 		mStancePosition = mSwingPosition;
	// 		mSwingPosition = T_swing;
	// 		// mStancePosition = T_stance;

	// 		// mSwingPosition.translation() -= R_ref*mCurrentHipPosition*0.8;
	// 		// mStancePosition.translation() += -R_ref*mCurrentHipPosition;

	// 		mR_IK0 = mR_IK1;
	// 		mP_IK0 = mP_IK1;

	// 		mR_IK1 = R;
	// 		mP_IK1 = mCurrentHipPosition;

	// 		mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, mR_IK1);
	// 		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, mR_IK1);
	// 		mphase = 0.0;
	// 	}
	// }



	// if((p_xz-mCurrentHipPosition).norm()>0.15 && mphase>1.0)
	// {
	// 	if((mSwing == 0 && p_xz[0]<0) || (mSwing == 1 && p_xz[0]>0))
	// 		mFootChanged = true;
	// 	else
	// 		mFootChanged = false;
	// 	mCurrentHipPosition = p_xz;
	// 	if(p_xz[0]>0)
	// 	{
	// 		mSwing = 0;
	// 		mStance = 1;
	// 	}
	// 	else
	// 	{
	// 		mSwing = 1;
	// 		mStance = 0;
	// 	}
	// 	int swing_foot = mSwing==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");	
	// 	int stance_foot = mSwing==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");

	// 	Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;
	// 	Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
	// 	Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];

	// 	mSwingPosition = T_swing;
	// 	mStancePosition = T_stance;


	// 	mSwingPosition.translation() += R_ref*mCurrentHipPosition*1.2;
	// 	mStancePosition.translation() += -R_ref*mCurrentHipPosition;
	// 	// mStancePosition.translation()[1] -= 0.1;
	// 	if(mR_IK1.rows()!=0){
	// 		mR_IK0 = mR_IK1;
	// 		mP_IK0 = mP_IK1;
	// 	}
	// 	else{
	// 		mP_IK0.setZero();
	// 		mR_IK0 = R;
	// 	}
	// 	mR_IK1 = R;
	// 	mP_IK1 = mCurrentHipPosition;

	// 	mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, mR_IK1);
	// 	mTwoJointIKs[mStance]->solve(mStancePosition, root_p, mR_IK1);
	// 	mphase = 0.0;
	// }
	// Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;
	
	if(mSolveFootIK)
	{
		Eigen::Vector3d root_p = baseP + R_ref*xT;
		if(mphase>1.0)
	{
		// if(mR_IK1.rows()!=0)
		// 	R = mR_IK1;
		mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, R);
		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, R);
	}
	else
	{
		double s = mphase;
		s = MotionUtils::easeInEaseOut(1.0-s, 0.0, 1.0);
		Eigen::Vector3d dir = xT - mP_IK0;
		root_p = baseP + (R_ref*(mP_IK0 + s*dir));
		s = 0.03*MotionUtils::easeInEaseOut(2*std::abs(s-0.5));
		root_p[1] += s;
		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, R);

		{
			int i0,i1,i2;
			if(mSwing == 0)
			{
				i0 = bvh->getNodeIndex("simLeftFoot");
				i1 = bvh->getNodeIndex("simLeftLeg");
				i2 = bvh->getNodeIndex("simLeftUpLeg");
			}
			else
			{
				i0 = bvh->getNodeIndex("simRightFoot");
				i1 = bvh->getNodeIndex("simRightLeg");
				i2 = bvh->getNodeIndex("simRightUpLeg");
			}
			Eigen::Matrix3d R00(mR_IK0.block<3,3>(0, i0*3)),R10(mR_IK0.block<3,3>(0, i1*3)),R20(mR_IK0.block<3,3>(0, i2*3));
			Eigen::Matrix3d R01(mR_IK1.block<3,3>(0, i0*3)),R11(mR_IK1.block<3,3>(0, i1*3)),R21(mR_IK1.block<3,3>(0, i2*3));
			Eigen::AngleAxisd aa0(R00.transpose()*R01);
			Eigen::AngleAxisd aa1(R10.transpose()*R11);
			Eigen::AngleAxisd aa2(R20.transpose()*R21);

			double s = mphase;
			s = MotionUtils::easeInEaseOut(1.0-s);

			aa0.angle() *= s;
			aa1.angle() *= s;
			aa2.angle() *= s;

			s = 0.6*MotionUtils::easeInEaseOut(2*std::abs(s-0.5));
			Eigen::AngleAxisd aa1x(s,Eigen::Vector3d::UnitX());
			Eigen::AngleAxisd aa2x(-s,Eigen::Vector3d::UnitX());
			R.block<3,3>(0, i0*3) = R00*(aa0.toRotationMatrix());
			R.block<3,3>(0, i1*3) = R10*(aa1.toRotationMatrix());
			R.block<3,3>(0, i2*3) = R20*(aa2.toRotationMatrix());

		}
	}
		mphase += mTimestep/mStepTime;

	return std::make_pair(root_p, R);
	}
	else
	{
		return std::make_pair(baseP, R);
	}
	


}

