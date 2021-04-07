#include "MassSpringDamperSystem.h"
#include "Motion.h"
#include "BVH.h"
#include "Character.h"
#include "TwoJointIK.h"
using namespace dart::dynamics;
MassSpringDamperSystem::
MassSpringDamperSystem(Character* character,
					const Eigen::VectorXd& mass_coeffs,
					const Eigen::VectorXd& spring_coeffs,
					const Eigen::VectorXd& damper_coeffs,
					double timestep)
	:mCharacter(character),mSkeleton(character->getSkeleton()), mNumJoints(character->getSkeleton()->getNumJoints()),mStepTime(0.5),
	mMassCoeffs(mass_coeffs*0.5),mSpringCoeffs(spring_coeffs*2.0),mDamperCoeffs(damper_coeffs),mTimestep(timestep) // FFF
{
	mR.resize(3,3*mNumJoints);	
	mw.resize(3,mNumJoints);
	mf.resize(3,mNumJoints);

	mPMassCoeffs = 50.0;
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
	}
	int cnt = 0;
	for(int i=0;i<mNumJoints;i++)
	{
		break;
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
	mPp[0] = std::max(-0.2,std::min(0.2,mPp[0]));
	mPp[2] = std::max(-0.2,std::min(0.2,mPp[2]));
	mPp += mTimestep*mPv;
	
	mf.setZero();
	mPf.setZero();

	Eigen::MatrixXd R(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		R.block<3,3>(0,i*3) = baseR.block<3,3>(0,i*3)*mR.block<3,3>(0,i*3);


	// Adjust root position and lower body
	Eigen::Isometry3d T_ref = mCharacter->getReferenceTransform();
	Eigen::Matrix3d R_ref = T_ref.linear();

	Eigen::Vector3d p_xz = mPp;
	p_xz[1] = 0.0;
	// std::cout<<mPp.transpose()<<std::endl;
	// mPp[1] = -0.3*p_xz.norm();

	// Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;

	if((p_xz-mCurrentHipPosition).norm()>0.15 && mphase>1.0)
	{
		if((mSwing == 0 && p_xz[0]<0) || (mSwing == 1 && p_xz[0]>0))
			mFootChanged = true;
		else
			mFootChanged = false;
		mCurrentHipPosition = p_xz;
		if(p_xz[0]>0)
		{
			mSwing = 0;
			mStance = 1;
		}
		else
		{
			mSwing = 1;
			mStance = 0;
		}
		int swing_foot = mSwing==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");	
		int stance_foot = mSwing==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");

		Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;
		Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
		Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];

		mSwingPosition = T_swing;
		mStancePosition = T_stance;


		mSwingPosition.translation() += R_ref*mCurrentHipPosition*1.2;
		mStancePosition.translation() += -R_ref*mCurrentHipPosition;
		// mStancePosition.translation()[1] -= 0.1;
		if(mR_IK1.rows()!=0){
			mR_IK0 = mR_IK1;
			mP_IK0 = mP_IK1;
		}
		else{
			mP_IK0.setZero();
			mR_IK0 = R;
		}
		mR_IK1 = R;
		mP_IK1 = mCurrentHipPosition;

		mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, mR_IK1);
		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, mR_IK1);
		mphase = 0.0;
	}
	Eigen::Vector3d root_p = baseP + R_ref*mCurrentHipPosition;
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
		Eigen::Vector3d dir = mCurrentHipPosition - mP_IK0;
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

			s = MotionUtils::easeInEaseOut(2*std::abs(s-0.5));
			Eigen::AngleAxisd aa1x(s,Eigen::Vector3d::UnitX());
			R.block<3,3>(0, i0*3) = R00*(aa0.toRotationMatrix());
			R.block<3,3>(0, i1*3) = R10*(aa1.toRotationMatrix());
			R.block<3,3>(0, i2*3) = R20*(aa2.toRotationMatrix());

		}
		if(mFootChanged)
		{
			int i0,i1,i2;
			if(mSwing == 1)
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

			s = 0.2*MotionUtils::easeInEaseOut(2*std::abs(s-0.5));
			Eigen::AngleAxisd aa1x(s,Eigen::Vector3d::UnitX());
			R.block<3,3>(0, i0*3) = R00*(aa0.toRotationMatrix());
			R.block<3,3>(0, i1*3) = R10*(aa1.toRotationMatrix());
			R.block<3,3>(0, i2*3) = R20*(aa2.toRotationMatrix());

		}
	}

	mphase += mTimestep/mStepTime;

	return std::make_pair(root_p, R);
}

