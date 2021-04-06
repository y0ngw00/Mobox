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
	:mCharacter(character),mSkeleton(character->getSkeleton()), mNumJoints(character->getSkeleton()->getNumJoints()),mStepTime(0.2),
	mMassCoeffs(mass_coeffs*0.5),mSpringCoeffs(spring_coeffs*2.0),mDamperCoeffs(damper_coeffs),mTimestep(timestep) // FFF
{
	mR.resize(3,3*mNumJoints);	
	mw.resize(3,mNumJoints);
	mf.resize(3,mNumJoints);

	mPMassCoeffs = 300.0;
	mPSpringCoeffs = 800.0;
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

	//Apply Twojoint IKs
	Eigen::Isometry3d T_ref = mCharacter->getReferenceTransform();
	Eigen::Matrix3d R_ref = T_ref.linear();

	Eigen::Vector3d p_xz = mPp;
	p_xz[1] = 0.0;
	mPp[1] = -0.1*p_xz.norm();
	
	Eigen::MatrixXd R(3,3*mNumJoints);
	for(int i=0;i<mNumJoints;i++)
		R.block<3,3>(0,i*3) = baseR.block<3,3>(0,i*3)*mR.block<3,3>(0,i*3);

	Eigen::Vector3d root_p = baseP+R_ref*mPp;

	if((p_xz-mCurrentHipPosition).norm()>0.1 && mphase>1.0)
	{
		mphase = 0.0;
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
		
		Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
		Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];
		mSwingPosition = T_swing;
		mStancePosition = T_stance;

		mSwingPosition.translation() += R_ref*(p_xz*1.5);
		mSwingPosition.translation()[1] += -mPp[1];

		mStancePosition.translation() += -R_ref*mPp;
		mCurrentHipPosition = p_xz;
		if(mR_IK1.rows()!=0)
			mR_IK0 = mR_IK1;
		else
			mR_IK0 = R;
		mR_IK1 = R;

		mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, mR_IK1);
	}
	if(mphase>1.0)
	{
		if(mR_IK1.rows()!=0)
			R = mR_IK1;
		mTwoJointIKs[mSwing]->solve(mSwingPosition, root_p, R);
		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, R);
	}
	else
	{
		mTwoJointIKs[mStance]->solve(mStancePosition, root_p, R);

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
		std::cout<<s<<std::endl;
		aa0.angle() *= s;
		aa1.angle() *= s;
		aa2.angle() *= s;

		

		s = 0.2*MotionUtils::easeInEaseOut(2*std::abs(s-0.5));
		Eigen::AngleAxisd aa1x(s,Eigen::Vector3d::UnitX());
		// Eigen::AngleAxisd aa2x(-s,Eigen::Vector3d::UnitX());
		R.block<3,3>(0, i0*3) = R00*(aa0.toRotationMatrix());
		R.block<3,3>(0, i1*3) = R10*(aa1.toRotationMatrix());//*(aa1x.toRotationMatrix());
		R.block<3,3>(0, i2*3) = R20*(aa2.toRotationMatrix());//*(aa2x.toRotationMatrix());

		R.block<3,3>(0, i0*3) = R01;
		R.block<3,3>(0, i1*3) = R11;
		R.block<3,3>(0, i2*3) = R21;
	}
	




	// if((p_xz-mCurrentHipPosition).norm()>0.1 && mphase>1.0)
	// {
	// 	mphase = 0.0;
	// 	mFoot = p_xz[0]>0?0:1;
	// 	mFootPosition = p_xz*1.5;
	// 	mCurrentHipPosition = p_xz;

		
	// 	int stance_foot = mFoot==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");
	// 	int swing_foot = mFoot==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");

	// 	mR_IK1 = mR_IK;
	// 	Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, mR_IK1, swing_foot)[0];
	// 	T_swing.translation() += R_ref*mFootPosition;
	// 	T_swing.translation()[1] -= mPp[1];
	// 	if(mFoot==0) mTwoJointIKs[0]->solve(T_swing, root_p, mR_IK1);
	// 	else mTwoJointIKs[1]->solve(T_swing, root_p, mR_IK1);
	// 	// for(int i=8;i<mNumJoints;i++)
	// 		// mR_IK1.block<3,3>(0,i*3) = mR_IK.block<3,3>(0,i*3).transpose()*mR_IK1.block<3,3>(0,i*3);
	// }

	// if(mphase>1.0)
	// {
	// 	int stance_foot = mFoot==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");
	// 	int swing_foot = mFoot==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");

	// 	Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, baseR, stance_foot)[0];
	// 	Eigen::Isometry3d T_swing = bvh->forwardKinematics(root_p, baseR, swing_foot)[0];
		
	// 	T_stance.translation() -= R_ref*mPp;
	// 	T_swing.translation() += R_ref*mFootPosition;
	// 	T_swing.translation()[1] -= mPp[1];
	// 	if(mFoot==0)
	// 	{
	// 		mTwoJointIKs[0]->solve(T_swing, root_p, R);
	// 		mTwoJointIKs[1]->solve(T_stance, root_p, R);
	// 	}
	// 	else
	// 	{
	// 		mTwoJointIKs[0]->solve(T_stance, root_p, R);
	// 		mTwoJointIKs[1]->solve(T_swing, root_p, R);
	// 	}
	// 	mR_IK = R;
	// }
	// else
	// {
	// 	int stance_foot = mFoot==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");

	// 	Eigen::MatrixXd R1 = mR_IK1;
	// 	Eigen::Isometry3d T_stance = bvh->forwardKinematics(root_p, R1, stance_foot)[0];
		
	// 	T_stance.translation() -= R_ref*mPp;

	// 	if(mFoot==0) mTwoJointIKs[1]->solve(T_stance, root_p, R1);
	// 	else mTwoJointIKs[0]->solve(T_stance, root_p, R1);


	// 	for(int i=8;i<mNumJoints;i++){
	// 		Eigen::AngleAxisd aa(R1.block<3,3>(0,i*3));
	// 		aa.angle() *= mphase;
	// 		// R.block<3,3>(0,i*3) = mR_IK.block<3,3>(0,i*3)*aa.toRotationMatrix();
	// 		R.block<3,3>(0,i*3) = R1.block<3,3>(0,i*3);
	// 	}
	// }



	// if(mphase<1.0)
	// {
	// 	Eigen::MatrixXd R1 = R;

	// 	int stance_foot = mFoot==0?bvh->getNodeIndex("simRightFoot"):bvh->getNodeIndex("simLeftFoot");
	// 	int swing_foot = mFoot==0?bvh->getNodeIndex("simLeftFoot"):bvh->getNodeIndex("simRightFoot");

	// 	Eigen::Isometry3d T_stance = bvh->forwardKinematics(baseP, baseR, stance_foot)[0];
	// 	Eigen::Isometry3d T_swing = bvh->forwardKinematics(baseP, baseR, swing_foot)[0];
		
	// 	T_swing.translation() += R_ref*mFootPosition;
	// 	if(mFoot==0)
	// 	{
	// 		mTwoJointIKs[0]->solve(T_swing, baseP+mPp, R1);
	// 		mTwoJointIKs[1]->solve(T_stance, baseP+mPp, R1);
	// 	}
	// 	else
	// 	{
	// 		mTwoJointIKs[0]->solve(T_stance, baseP+mPp, R1);
	// 		mTwoJointIKs[1]->solve(T_swing, baseP+mPp, R1);
	// 	}
	// 	// for(int i=0;i<mNumJoints;i++){
	// 	// 	Eigen::Matrix3d R01 = R.block<3,3>(0,i*3).transpose()*R1.block<3,3>(0,i*3);
	// 	// 	Eigen::AngleAxisd aa01(R01);
	// 	// 	aa01.angle() *= mphase;

	// 	// 	R.block<3,3>(0,i*3) = R.block<3,3>(0,i*3)*aa01.toRotationMatrix();
	// 	// }
	// 	R = R1;
	// }

	mphase += mTimestep/mStepTime;



	return std::make_pair(root_p, R);
}

