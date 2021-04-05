#include <functional>
#include <fstream>
#include <sstream>
#include "Environment.h"
#include "DARTUtils.h"
#include "MassSpringDamperSystem.h"
#include "MathUtils.h"
#include "Event.h"
#include "Character.h"
#include "BVH.h"
#include "Motion.h"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/fcl/fcl.hpp"

using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
Environment::
Environment()
	:mControlHz(30),
	mSimulationHz(300),
	mElapsedFrame(0),
	mStartFrame(0),
	mCurrentFrame(0),
	mMaxElapsedFrame(300),
	mWorld(std::make_shared<World>()),
	mEvent(nullptr),
	mState0Dirty(true),
	mState1Dirty(true),
	mKinematic(false)
{
	dart::math::Random::generateSeed(true);
	mSimCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel_c.xml");
	mKinCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel_c.xml");

	BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/open_door.bvh");
	Motion* motion = new Motion(bvh);
	// if(0)
	// {
	// Motion* non_cyclic_motion = new Motion(bvh);

	// for(int i=100;i<325;i++)
	// 	non_cyclic_motion->append(bvh->getPosition(i),bvh->getRotation(i),false);

	
	

	
	// Eigen::MatrixXd D = MotionUtils::computePoseDifferences(non_cyclic_motion);

	// int s0=10,s1=70;
	// int e0=110,e1=160;
	// int s,e;
	// double min_diff = 1e6;
	// D.block(s0,e0,s1-s0,e1-e0).minCoeff(&s,&e);
	// s += s0;
	// e += e0;
	
	// Eigen::MatrixXd disp_pos = bvh->getPosition(e) - bvh->getPosition(s);
	// Eigen::MatrixXd disp = MotionUtils::computePoseDisplacement(bvh->getRotation(s), bvh->getRotation(e));

	
	// for(int i=0;i<10;i++)
	// 	for(int j=s;j<e;j++){
	// 		if(j-s<15)
	// 		{
	// 			double alpha = MotionUtils::easeInEaseOut((j-s)/15.0);

	// 			motion->append(bvh->getPosition(j)+alpha*disp_pos, MotionUtils::addDisplacement(bvh->getRotation(j), alpha*disp),false);
	// 		}
	// 		else
	// 			motion->append(bvh->getPosition(j),bvh->getRotation(j),false);
	// 	}
	// }
	int nf = bvh->getNumFrames();
	for(int i=0;i<nf;i++)
		motion->append(bvh->getPosition(i), bvh->getRotation(i), false);
	for(int i=0;i<mMaxElapsedFrame-nf+50;i++)
		motion->append(bvh->getPosition(nf-1), bvh->getRotation(nf-1), false);
	motion->computeVelocity();
	std::cout<<motion->getNumFrames()<<std::endl;

	mSimCharacter->buildBVHIndices(bvh->getNodeNames());
	mKinCharacter->buildBVHIndices(bvh->getNodeNames());
	mSimCharacter->setBaseMotionAndCreateMSDSystem(motion);

	mGround = DARTUtils::createGround(1.0);

	{
		mObstacles.emplace_back(DARTUtils::createBox(1000.0, Eigen::Vector3d(0.5,1.5,0.5),"Weld"));
		Eigen::Isometry3d T_w = Eigen::Isometry3d::Identity();
		std::ifstream ifs(std::string(ROOT_DIR)+"/temp.txt");
		double x,y,z;
		ifs>>x>>y>>z;
		ifs.close();
		T_w.translation() = Eigen::Vector3d(-2.0, 2.0,-0.5);
		mObstacles.back()->getJoint(0)->setTransformFromParentBodyNode(T_w);
		mWorld->addSkeleton(mObstacles.back());

	}
	

	{
		mObstacles.emplace_back(DARTUtils::createBox(1000.0, Eigen::Vector3d(0.5,1.5,0.5),"Weld"));
		Eigen::Isometry3d T_w = Eigen::Isometry3d::Identity();
		std::ifstream ifs(std::string(ROOT_DIR)+"/temp.txt");
		double x,y,z;
		ifs>>x>>y>>z;
		ifs.close();
		T_w.translation() = Eigen::Vector3d(x, y, z);
		mObstacles.back()->getJoint(0)->setTransformFromParentBodyNode(T_w);

		mWorld->addSkeleton(mObstacles.back());
	}
	
	
	mWashWindow = DARTUtils::createBox(1000.0,Eigen::Vector3d(2.0, 2.0, 0.1),"Weld");
	Eigen::Isometry3d T_w = Eigen::Isometry3d::Identity();
	T_w.translation() = Eigen::Vector3d(0.0, 2.0, 0.8);
	mWashWindow->getJoint(0)->setTransformFromParentBodyNode(T_w);
	double friction = 0.001;
	mWashWindow->getBodyNode(0)->setFrictionCoeff(friction);
	mSimCharacter->getSkeleton()->getBodyNode("RightHand")->setFrictionCoeff(friction);
	mSimCharacter->getSkeleton()->getBodyNode("RightForeArm")->setFrictionCoeff(0.05);
	mWorld->addSkeleton(mWashWindow);
	
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

	mWorld->addSkeleton(mSimCharacter->getSkeleton());
	mWorld->addSkeleton(mGround);
	mWorld->setTimeStep(1.0/(double)mSimulationHz);
	mWorld->setGravity(Eigen::Vector3d(0,-9.81,0.0));

	mImitationFrameWindow.emplace_back(0);
	mImitationFrameWindow.emplace_back(1);
	// mImitationFrameWindow.emplace_back(5);

	mWeightPos = 4.0;
	mWeightVel = 0.1;
	mWeightEE = 1.0;
	mWeightRoot = 1.0;
	mWeightCOM = 1.0;

	mRewards.insert(std::make_pair("r_pos", std::vector<double>()));
	mRewards.insert(std::make_pair("r_vel", std::vector<double>()));
	mRewards.insert(std::make_pair("r_ee", std::vector<double>()));
	mRewards.insert(std::make_pair("r_root", std::vector<double>()));
	mRewards.insert(std::make_pair("r_com", std::vector<double>()));
	mRewards.insert(std::make_pair("r_imit", std::vector<double>()));
	mRewards.insert(std::make_pair("r_force", std::vector<double>()));
	mRewards.insert(std::make_pair("r_force_orig", std::vector<double>()));
	mRewards.insert(std::make_pair("r", std::vector<double>()));

	int n = motion->getNumFrames() - mMaxElapsedFrame - 10;
	int stride = 30;

	mInitialStateDistribution = new Distribution1D<int>(n/stride, [=](int i)->int{return i*stride;}, 0.7);
	// mInitialStateDistribution = new Distribution1D<int>(1, [=](int i){return i;}, 0.7);
	mStartFrame = mInitialStateDistribution->sample();
	
	// mHapticDistribution = new Distribution1D<double>(8,[=](int i)->double{std::vector<double> x={0.0, 0.01, 0.02,0.05,0.1,0.12,0.15,0.2};return x[i];}, 0.7);
	mHapticDistribution = new Distribution1D<double>(2,[=](int i)->double{std::vector<double> x={0.0, 0.1};return x[i];}, 0.7);
	mHapticBoundary =0.0;


	// mTargetHandPositions = new Distribution2D<Eigen::Vector3d>(16,8,[=](int i, int j)->Eigen::Vector3d{

	// })
	// mTargetHandPositionDistribution

	



	mEvent = new ObstacleEvent(this, mSimCharacter);
	this->reset();
	
	mR = 0.4;
	mTHETA = 1.0;
	mPHI = 3.0;

	// mR = dart::math::Random::uniform<double>(0.3, 0.5);
	// mTHETA = dart::math::Random::uniform<double>(0.0,0.5*M_PI);
	// mPHI = dart::math::Random::uniform<double>(0.0,2*M_PI);

	mActionSpace0 = this->getActionSpace0();
	mActionWeight0 = this->getActionWeight0();
	mActionSpace1 = this->getActionSpace1();
	mActionWeight1 = this->getActionWeight1();
}


void
Environment::
reset(int frame)
{

	// mR = 0.4;
	// mTHETA = 1.0;
	// mPHI = 2.3;

	// mR += dart::math::Random::uniform<double>(-0.1, 0.1);
	// mTHETA += dart::math::Random::uniform<double>(-0.5,0.5);
	// mPHI += dart::math::Random::uniform<double>(-0.5, 0.5);

	// this->setRandomTargetCenter();

	mElapsedFrame = 0;
	mPrevAction1 = Eigen::Vector3d::UnitZ();
	Eigen::Map<Eigen::ArrayXd> rs(mRewards["r"].data(), mRewards["r"].size());
	// Eigen::Map<Eigen::ArrayXd> rfs(mRewards["r_force"].data(), mRewards["r_force"].size());
	mInitialStateDistribution->update(rs.sum());
	if(frame<0)
		mStartFrame = mInitialStateDistribution->sample();
	else
		mStartFrame = frame;
	// mHapticDistribution->update(rfs.sum());
	// mHapticBoundary = mHapticDistribution->sample();

	mCurrentFrame = mStartFrame;
	mEvent->reset();
	mSimCharacter->resetMotion(mStartFrame);

	mSimCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
						mSimCharacter->getRotation(mCurrentFrame),
						mSimCharacter->getLinearVelocity(mCurrentFrame),
						mSimCharacter->getAngularVelocity(mCurrentFrame));
	mSimCharacter->getSkeleton()->clearConstraintImpulses();
	mSimCharacter->getSkeleton()->clearInternalForces();
	mSimCharacter->getSkeleton()->clearExternalForces();
	auto fss = mSimCharacter->getForceSensors();
	for(auto fs: fss)
		fs->reset();

	mKinCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
						mSimCharacter->getRotation(mCurrentFrame),
						mSimCharacter->getLinearVelocity(mCurrentFrame),
						mSimCharacter->getAngularVelocity(mCurrentFrame));
	for(auto r : mRewards){
		mRewards[r.first].clear();
		mRewards[r.first].reserve(mMaxElapsedFrame);
	}
	mState0Dirty = true;
	mState1Dirty = true;


}

void
Environment::
step(const Eigen::VectorXd& _action0, const Eigen::VectorXd& _action1)
{
	double alpha = dart::math::Random::uniform<double>(0.0,1.0);
	
	Eigen::VectorXd action0 = this->convertToRealActionSpace0(_action0);
	Eigen::VectorXd action1 = this->convertToRealActionSpace1(_action1);
	mCurrAction1 = action1;
	int num_sub_steps = mSimulationHz/mControlHz;
	// action_force.setZero();

	// 	Eigen::Vector3d center = Eigen::Vector3d(-0.0003061, 2.44461, 0.534614);

	// double x = mR*std::sin(mTHETA)*std::cos(mPHI);
	// double y = mR*std::sin(mTHETA)*std::sin(mPHI);
	// double z = mR*std::cos(mTHETA);

	// center[0] += x;
	// center[1] += y;
	// center[2] += z;

	// center = center - mSimCharacter->getForceSensors()[0]->getPosition();
	// action1 = center*1000.0;

	mSimCharacter->stepMotion(action1);
	mKinCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
							mSimCharacter->getRotation(mCurrentFrame),
							mSimCharacter->getLinearVelocity(mCurrentFrame),
							mSimCharacter->getAngularVelocity(mCurrentFrame));
	Eigen::MatrixXd base_rot = mSimCharacter->getRotation(mCurrentFrame);
	Eigen::MatrixXd base_ang_vel = mSimCharacter->getAngularVelocity(mCurrentFrame);

	auto target_pv = mSimCharacter->computeTargetPosAndVel(base_rot, action0, base_ang_vel);
	mEvent->call();
	auto fss = mSimCharacter->getForceSensors();
	auto obstacle = dynamic_cast<ObstacleEvent*>(mEvent)->getObstacle();
	Eigen::Vector6d p_o = Eigen::compose(Eigen::Vector3d::Zero(),dynamic_cast<ObstacleEvent*>(mEvent)->getLinearPosition());
	Eigen::Vector6d v_o = Eigen::compose(Eigen::Vector3d::Zero(),dynamic_cast<ObstacleEvent*>(mEvent)->getLinearVelocity());

	// int xx = 10; //XXX
	for(int i=0;i<num_sub_steps;i++)
	{
		mSimCharacter->actuate(target_pv.first, target_pv.second);
		if(mKinematic)
		mSimCharacter->setPose(mSimCharacter->getPosition(mCurrentFrame),
								mSimCharacter->getRotation(mCurrentFrame),
								mSimCharacter->getLinearVelocity(mCurrentFrame),
								mSimCharacter->getAngularVelocity(mCurrentFrame));
		obstacle->setPositions(p_o);
		obstacle->setVelocities(v_o);
		if(!mKinematic) mWorld->step();	
		auto cr = mWorld->getConstraintSolver()->getLastCollisionResult();

		// for(int j=0;j<xx;j++)//XXX
		// {//XXX


		for (auto j = 0u; j < cr.getNumContacts(); ++j)
		{
			auto contact = cr.getContact(j);

			auto shapeFrame1 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject1->getShapeFrame());
			auto shapeFrame2 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject2->getShapeFrame());

			std::string name1 = shapeFrame1->asShapeNode()->getBodyNodePtr()->getSkeleton()->getName();
			std::string name2 = shapeFrame2->asShapeNode()->getBodyNodePtr()->getSkeleton()->getName();

			if(name1 == "ground" or name2 == "ground")
				continue;

			Eigen::Vector3d force = contact.force;
			if(name2 == "humanoid")
				force = -force;

			auto fs = mSimCharacter->getClosestForceSensor(contact.point);
			if(fs!=nullptr)
				fs->addExternalForce(force);
		}
		for(auto fs: fss)
			fs->step();
		// } //XXX
	}

	mElapsedFrame++;
	mCurrentFrame = mStartFrame + mElapsedFrame;
	mState0Dirty = true, mState1Dirty = true;


	
}

Eigen::VectorXd
Environment::
getState0()
{
	if(mState0Dirty == false)
		return mState0;
	std::vector<Eigen::VectorXd> state;

	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
	Eigen::Matrix3d R_sim_inv = T_sim_inv.linear();
	
	std::vector<Eigen::Vector3d> state_sim = mSimCharacter->getState();
	int n = state_sim.size();

	state.emplace_back(MathUtils::ravel(state_sim));
	Eigen::VectorXd state_kin_save = mKinCharacter->saveState();
	for(int i=0;i<mImitationFrameWindow.size();i++)
	{
		int frame = mCurrentFrame + mImitationFrameWindow[i];
		mKinCharacter->setPose(mSimCharacter->getPosition(frame),
						mSimCharacter->getRotation(frame),
						mSimCharacter->getLinearVelocity(frame),
						mSimCharacter->getAngularVelocity(frame));

		Eigen::Isometry3d T_kin = mKinCharacter->getReferenceTransform();
		std::vector<Eigen::Vector3d> state_kin = mKinCharacter->getState();
		
		std::vector<Eigen::Vector3d> state_sim_kin_diff(n+2);
		Eigen::Isometry3d T_sim_kin_diff = T_sim_inv*T_kin;
		for(int j=0;j<n;j++)
			state_sim_kin_diff[j] = state_sim[j] - state_kin[j];

		state_sim_kin_diff[n+0] = T_sim_kin_diff.translation();
		state_sim_kin_diff[n+1] = T_sim_kin_diff.linear().col(2);

		state.emplace_back(MathUtils::ravel(state_kin));
		state.emplace_back(MathUtils::ravel(state_sim_kin_diff));
	}

	mKinCharacter->restoreState(state_kin_save);
	Eigen::VectorXd s = MathUtils::ravel(state);

	mState0 = s;
	mState0Dirty = false;
	return s;
}
Eigen::VectorXd
Environment::
getState1()
{
	Eigen::VectorXd state0 = this->getState0();
	if(mState1Dirty==false){
		Eigen::VectorXd s(mState0.rows() + mState1.rows());
		s<<state0,mState1;
		// return s;

		return mState1;
	}

	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
	Eigen::Matrix3d R_sim_inv = T_sim_inv.linear();

	auto fss = mSimCharacter->getForceSensors();
	auto statefs = mSimCharacter->getStateForceSensors();

	const auto& ps = statefs["ps"];
	const auto& vs = statefs["vs"];
	const auto& hps = statefs["hps"];
	const auto& hvs = statefs["hvs"];
		
	int m = ps.cols();
	// Eigen::MatrixXd state(m, 12);
	std::vector<Eigen::VectorXd> state;

	for(int i=0;i<m;i++){
		state.emplace_back(ps.col(i));
		state.emplace_back(vs.col(i));
		state.emplace_back(hps.col(i));
		state.emplace_back(hvs.col(i));
	}

	// Eigen::Vector3d center = this->getTargetCenter();
	
	// // center -= T_sim*((Eigen::Vector3d)ps.col(0));
	// center = T_sim_inv*center;
	// // std::cout<<center.transpose()<<std::endl;
	// state.emplace_back(center);
	// state.emplace_back(center-(Eigen::Vector3d)ps.col(0));
	// state.emplace_back(mPrevAction1);
	mState1 = MathUtils::ravel(state);
	mState1Dirty = false;
	// std::cout<<mState1.transpose()<<std::endl;


	Eigen::VectorXd s(state0.rows() + mState1.rows());
	s<<state0,mState1;
	// return s;
	return mState1;
}

std::map<std::string,double>
Environment::
getReward()
{
	Eigen::VectorXd joint_weights = mSimCharacter->getJointWeights();

	std::map<std::string,double> rewards;

	auto state_sim_body = mSimCharacter->getStateBody();
	auto state_sim_joint = mSimCharacter->getStateJoint();
	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();
	Eigen::Isometry3d T_sim_inv = T_sim.inverse();
	Eigen::Vector3d sim_com = mSimCharacter->getSkeleton()->getCOM();
	Eigen::Vector3d sim_com_vel = mSimCharacter->getSkeleton()->getCOMLinearVelocity();
	Eigen::MatrixXd sim_body_p, sim_body_R, sim_body_v, sim_body_w, sim_joint_p, sim_joint_v;
	
	sim_body_p = state_sim_body["ps"];
	sim_body_R = state_sim_body["Rs"];
	sim_body_v = state_sim_body["vs"];
	sim_body_w = state_sim_body["ws"];

	sim_joint_p = state_sim_joint["p"];
	sim_joint_v = state_sim_joint["v"];

	auto state_kin_body = mKinCharacter->getStateBody();
	auto state_kin_joint = mKinCharacter->getStateJoint();
	Eigen::Isometry3d T_kin = mKinCharacter->getReferenceTransform();
	Eigen::Isometry3d T_kin_inv = T_kin.inverse();
	Eigen::Vector3d kin_com = mKinCharacter->getSkeleton()->getCOM();
	Eigen::Vector3d kin_com_vel = mKinCharacter->getSkeleton()->getCOMLinearVelocity();
	Eigen::MatrixXd kin_body_p, kin_body_R, kin_body_v, kin_body_w, kin_joint_p, kin_joint_v;
	
	kin_body_p = state_kin_body["ps"];
	kin_body_R = state_kin_body["Rs"];
	kin_body_v = state_kin_body["vs"];
	kin_body_w = state_kin_body["ws"];

	kin_joint_p = state_kin_joint["p"];
	kin_joint_v = state_kin_joint["v"];

	int n = mSimCharacter->getSkeleton()->getNumBodyNodes();

	double error_pos=0.0, error_vel=0.0, error_ee=0.0, error_root=0.0, error_com=0.0;
	// pos error
	Eigen::MatrixXd diff_pos = DARTUtils::computeDiffPositions(sim_joint_p, kin_joint_p);
	Eigen::MatrixXd diff_vel = sim_joint_v - kin_joint_v;
	Eigen::MatrixXd lb_vel = Eigen::MatrixXd::Constant(diff_vel.rows(),diff_vel.cols(), -1.0);
	Eigen::MatrixXd ub_vel = Eigen::MatrixXd::Constant(diff_vel.rows(),diff_vel.cols(), 1.0);
	diff_vel = diff_vel.cwiseMax(lb_vel).cwiseMin(ub_vel);

	for(int i=1;i<n;i++)
	{
		error_pos += joint_weights[i]*(diff_pos.col(i).dot(diff_pos.col(i)));
		error_vel += joint_weights[i]*(diff_vel.col(i).dot(diff_vel.col(i)));
	}

	auto ees = mSimCharacter->getEndEffectors();
	for(int i=0;i<ees.size();i++)
	{
		int idx = ees[i]->getIndexInSkeleton();
		Eigen::Vector3d sim_ee = sim_body_p.col(idx);
		Eigen::Vector3d kin_ee = kin_body_p.col(idx);
		
		Eigen::Vector3d diff_ee_local = T_sim_inv*sim_ee - T_kin_inv*kin_ee;
		error_ee += diff_ee_local.dot(diff_ee_local);
	}

	Eigen::Vector3d diff_root_p = sim_body_p.col(0) - kin_body_p.col(0);
	Eigen::Vector3d diff_root_R = dart::math::logMap(sim_body_R.block<3,3>(0,0).transpose()*kin_body_R.block<3,3>(0,0));
	Eigen::Vector3d diff_root_v = sim_body_v.col(0) - kin_body_v.col(0);
	Eigen::Vector3d diff_root_w = sim_body_w.col(0) - kin_body_w.col(0);

	error_root = 1.0 * diff_root_p.dot(diff_root_p) + 
				0.1* diff_root_R.dot(diff_root_R) + 
				0.01* diff_root_v.dot(diff_root_v) + 
				0.01* diff_root_w.dot(diff_root_w);

	Eigen::Vector3d diff_com = T_sim_inv*sim_com - T_kin_inv*kin_com;
	Eigen::Vector3d diff_com_vel = T_sim_inv.linear()*sim_com_vel - T_kin_inv.linear()*kin_com_vel;
	
	error_com = 1.0*diff_com.dot(diff_com) + 
				0.1*diff_com_vel.dot(diff_com_vel);

	double r_pos = std::exp(-mWeightPos*error_pos);
	double r_vel = std::exp(-mWeightVel*error_vel);
	double r_ee = std::exp(-mWeightEE*error_ee);
	double r_root = std::exp(-mWeightRoot*error_root);
	double r_com = std::exp(-mWeightCOM*error_com);

	r_root = std::max(0.5, r_root);

	Eigen::VectorXd applied_forces = mSimCharacter->getAppliedForces();

	int idx_hip = mSimCharacter->getSkeleton()->getJoint("Spine1")->getIndexInSkeleton(0);
	int idx_sl = mSimCharacter->getSkeleton()->getJoint("LeftArm")->getIndexInSkeleton(0);
	int idx_sr = mSimCharacter->getSkeleton()->getJoint("RightArm")->getIndexInSkeleton(0);
	int idx_hl = mSimCharacter->getSkeleton()->getJoint("LeftUpLeg")->getIndexInSkeleton(0);
	int idx_hr = mSimCharacter->getSkeleton()->getJoint("RightUpLeg")->getIndexInSkeleton(0);

	Eigen::Vector3d fhip = applied_forces.segment<3>(idx_hip);
	Eigen::Vector3d fsl = applied_forces.segment<3>(idx_sl);
	Eigen::Vector3d fsr = applied_forces.segment<3>(idx_sr);
	Eigen::Vector3d fhl = applied_forces.segment<3>(idx_hl);
	Eigen::Vector3d fhr = applied_forces.segment<3>(idx_hr);

	Eigen::Vector3d force_boundary = Eigen::Vector3d::Constant(20.0);

	Eigen::Vector3d ofhip = (fhip.cwiseAbs() - force_boundary).cwiseMax(Eigen::Vector3d::Zero());
	Eigen::Vector3d ofsl = (fsl.cwiseAbs() - force_boundary).cwiseMax(Eigen::Vector3d::Zero());
	Eigen::Vector3d ofsr = (fsr.cwiseAbs() - force_boundary).cwiseMax(Eigen::Vector3d::Zero());
	Eigen::Vector3d ofhl = (fhl.cwiseAbs() - force_boundary).cwiseMax(Eigen::Vector3d::Zero());
	Eigen::Vector3d ofhr = (fhr.cwiseAbs() - force_boundary).cwiseMax(Eigen::Vector3d::Zero());

	auto fss = mSimCharacter->getForceSensors();
	auto statefs = mSimCharacter->getStateForceSensors();

	const auto& ps = statefs["ps"];
	const auto& vs = statefs["vs"];
	const auto& hps = statefs["hps"];
	const auto& hvs = statefs["hvs"];

	// double error_force = 0.0;
	// int m = ps.cols();
	// double haptic_boundary = 1e-1;
	// double hps_norm = hps.colwise().norm().sum();
	// error_force = std::abs(mHapticBoundary - hps_norm);
	// // Eigen::MatrixXd bounded_ps = (haptic_boundary - hps.cwiseAbs()).cwiseMax(Eigen::MatrixXd::Zero(3, m));
	// // std::cout<<(haptic_boundary - hps_norm).transpose()<<std::endl;
	// // double r_force = std::exp(-5.0*bounded_ps.norm());
	// // std::cout<<haptic_boundary - hps_norm<<std::endl;
	// const Eigen::MatrixXd& msd_R = mSimCharacter->getMSDSystem()->getR();
	// const Eigen::MatrixXd& msd_w = mSimCharacter->getMSDSystem()->getw();

	// Eigen::VectorXd msd_angle = Eigen::VectorXd::Zero(n);
	// error_pos = 0.0;
	// for(int i=0;i<n;i++)
	// {
	// 	Eigen::Vector3d r = dart::math::logMap(msd_R.block<3,3>(0,i*3));
	// 	error_pos += joint_weights[i]*r.dot(r);
	// }
	
	// Eigen::Vector3d center = this->getTargetCenter();

	// // error_force = (center - T_sim*((Eigen::Vector3d)ps.col(28))).squaredNorm();
	// Eigen::Vector3d fp = T_sim*((Eigen::Vector3d)ps.col(0));
	// Eigen::Vector3d fv = T_sim.linear()*((Eigen::Vector3d)vs.col(0));
	// __fv = T_sim.linear()*((Eigen::Vector3d)vs.col(0));
	// Eigen::Vector3d dir = center - fp;
	// dir.normalize();

	// double error_force_v = std::max(0.0, 0.3-fv.dot(dir));
	// double error_force_p = (center - T_sim*((Eigen::Vector3d)ps.col(0))).squaredNorm();
	// // std::cout<<fv.dot(dir)<<std::endl;
	// // double r_force = std::exp(-10.0*error_force);
	// double r_force = std::exp(-mWeightEE*20.0*error_force_p);
	// r_force = std::max(1.0-2.0*std::cbrt(error_force_p), 0.0);
	// double r_force_orig = r_force;
	// // if(mRewards["r_force_orig"].size()>0)
	// // {
	// // 	double prev_r = mRewards["r_force_orig"].back();
	// // 	if(prev_r<r_force)
	// // 		r_force *= 1.5;
	// // 	else
	// // 		r_force *= 0.5;
	// // }
	// double r_diff = std::exp(-mWeightPos*error_pos);
	// r_diff = 1.0;
	// // std::cout<<(center - T_sim*((Eigen::Vector3d)ps.col(0))).transpose()<<std::endl;

	// double error_action = 0.0;
	// {
	// 	Eigen::Isometry3d T_sim = mSimCharacter->getReferenceTransform();

	// 	auto statefs = mSimCharacter->getStateForceSensors();

	// 	const auto& ps = statefs["ps"];
	// 	const auto& vs = statefs["vs"];
	// 	const auto& hps = statefs["hps"];
	// 	const auto& hvs = statefs["hvs"];

	// 	Eigen::Vector3d fp = T_sim*((Eigen::Vector3d)ps.col(0));
	// 	Eigen::Vector3d center = this->getTargetCenter();
	// 	if((fp-center).norm()<1e-1)
	// 	{
	// 		this->setRandomTargetCenter();
	// 	}
	// }
	// // if(mPrevAction1.norm()>1e-6)
	// // {
	// // 	Eigen::AngleAxisd aa_action(Eigen::Quaterniond::FromTwoVectors(mPrevAction1, mCurrAction1));
	// // 	double action_diff = aa_action.angle();
	// // 	error_action = std::max(action_diff-0.3, 0.0);
	// // 	// error_action = std::max(0.1-action_diff, 0.0);
		
	// // }
	// double r_action = std::exp(-1.0*error_action);
	// r_action = 1.0;
	// // std::cout<<r_force*r_action<<std::endl;
	// mPrevAction1 = mCurrAction1;
	// if(mElapsedFrame<30){
	// 	double alpha = (30 - mElapsedFrame)/30.0;
	// 	r_force = alpha + (1-alpha)*r_force;
	// }
	// std::cout<<mHapticBoundary<<" "<<hps_norm<<" "<<r_force<<" "<<r_diff<<std::endl;

	// if(this->isSleepf()==false)
	// std::cout<<"r_force  "<<r_force<<std::endl;
	// std::cout<<std::endl;
	// std::cout<<r_force<<std::endl;
	// for(int i=0;i<m;i++){
	// 	Eigen::Vector3d bounded_ps = (hps.col(i).cwiseAbs() - force_boundary).cwiseMax(Eigen::Vector3d::Zero())
	// 	error_force += hps.col(i).norm();
	// }
	// double error_force = ofsl.norm()+ofsr.norm()+ofhl.norm()+ofhr.norm();
	// double error_force = ofhip.norm() + ofsl.norm() + ofsr.norm();

	// std::cout<<fsl.cwiseAbs().norm()<<std::endl<<std::endl;
	// std::cout<<error_force<<std::endl;
	// if(mElapsedFrame<10)
	// 	error_force = 0.0;

	// double r_force = std::exp(-0.2*error_force);
	// std::cout<<r_force<<std::endl;
	double r_imit = r_pos*r_vel*r_ee*r_root*r_com;
	rewards.insert(std::make_pair("r_pos",r_pos));
	rewards.insert(std::make_pair("r_vel",r_vel));
	rewards.insert(std::make_pair("r_ee",r_ee));
	rewards.insert(std::make_pair("r_root",r_root));
	rewards.insert(std::make_pair("r_com",r_com));
	// rewards.insert(std::make_pair("r_force",r_force*r_diff*r_action));
	rewards.insert(std::make_pair("r_force",1.0));
	rewards.insert(std::make_pair("r_force_orig",1.0));
	rewards.insert(std::make_pair("r_imit",r_imit));
	rewards.insert(std::make_pair("r",r_imit));

	for(auto rew : rewards)
	{
		if(dart::math::isNan(rew.second))
			rewards[rew.first] = -30.0;
		mRewards[rew.first].emplace_back(rewards[rew.first]);
	}

	return rewards;
}
bool
Environment::
isSleep()
{
	for(auto fs: mSimCharacter->getForceSensors())
		if(!fs->isSleep())
			return false;

	return true;
}

bool
Environment::
inspectEndOfEpisode()
{
	double r_mean = 0.0;
	{
		const std::vector<double>& rs = mRewards["r_imit"];
		int n = rs.size();
		for(int i=std::max(0,n-30);i<n;i++){
			r_mean += rs[i];
		}
		for(int i=n-30;i<0;i++)
			r_mean += 1.0;
		r_mean /= 30.0;
	}

	double r_force_mean = 0.0;
	{
		const std::vector<double>& rs = mRewards["r_force"];
		int n = rs.size();
		for(int i=std::max(0,n-30);i<n;i++){
			r_force_mean += rs[i];
		}
		for(int i=n-30;i<0;i++)
			r_force_mean += 1.0;
		r_force_mean /= 30.0;
		// std::cout<<r_force_mean<<std::endl;
	}
	if(mElapsedFrame>=mMaxElapsedFrame){
		return true;
	}
	else if(r_mean<0.1){
		return true;
	}
	else if(r_force_mean<0.1){
		return true;
	}
	return false;
}
int
Environment::
getDimState0()
{
	return this->getState0().rows();
}
int
Environment::
getDimAction0()
{
	int n = mSimCharacter->getSkeleton()->getNumDofs();
	return n-6;
}
int
Environment::
getDimState1()
{
	return this->getState1().rows();
}

int
Environment::
getDimAction1()
{
	// return mSimCharacter->getMSDSystem()->getNumDofs()*mSimCharacter->getForceSensors().size(); //PPP
	return 3*mSimCharacter->getForceSensors().size(); //FFF
	// return 3*mSimCharacter->getForceSensors().size() + mSimCharacter->getMSDSystem()->getNumDofs(); //FFF


}

Eigen::MatrixXd
Environment::
getActionSpace0()
{
	Eigen::MatrixXd action_space = Eigen::MatrixXd::Ones(this->getDimAction0(), 2);
	action_space.col(0) *= -M_PI; // Lower
	action_space.col(1) *=  M_PI; // Upper

	return action_space;
}
Eigen::VectorXd
Environment::
getActionWeight0()
{
	Eigen::VectorXd action_weight = Eigen::VectorXd::Ones(this->getDimAction0());
	action_weight *= 0.3;
	return action_weight;
}
Eigen::VectorXd
Environment::
convertToRealActionSpace0(const Eigen::VectorXd& a_norm)
{
	Eigen::VectorXd a_real;
	Eigen::VectorXd lo = mActionSpace0.col(0), hi =  mActionSpace0.col(1);
	a_real = dart::math::clip<Eigen::VectorXd, Eigen::VectorXd>(a_norm, lo, hi);
	a_real = mActionWeight0.cwiseProduct(a_real);
	return a_real;
}
Eigen::MatrixXd
Environment::
getActionSpace1()
{
	Eigen::MatrixXd action_space = Eigen::MatrixXd::Ones(this->getDimAction1(), 2);
	int n = mSimCharacter->getMSDSystem()->getNumDofs();
	int m = 3*mSimCharacter->getForceSensors().size();

	action_space.col(0) *= -1.0;
	action_space.col(1) *= 1.0;
	return action_space;
}
Eigen::VectorXd
Environment::
getActionWeight1()
{
	Eigen::VectorXd action_weight = Eigen::VectorXd::Ones(this->getDimAction1());
	int n = mSimCharacter->getMSDSystem()->getNumDofs();
	int m = 3*mSimCharacter->getForceSensors().size();

	action_weight.head(m) *= 1000.0; //FFF
	// action_weight.tail(n) *= 0.8; //FFF

	//action_weight *= 0.3; //PPP
	return action_weight;
}
Eigen::VectorXd
Environment::
convertToRealActionSpace1(const Eigen::VectorXd& a_norm)
{
	Eigen::VectorXd a_real;
	Eigen::VectorXd lo = mActionSpace1.col(0), hi =  mActionSpace1.col(1);
	a_real = dart::math::clip<Eigen::VectorXd, Eigen::VectorXd>(a_norm, lo, hi);
	a_real = mActionWeight1.cwiseProduct(a_real);
	
	int n = mSimCharacter->getMSDSystem()->getNumDofs();
	int m = 3*mSimCharacter->getForceSensors().size();

	// a_real = 0.7*mPrevAction1 + 0.3*a_real;

	// Eigen::AngleAxisd aa(Eigen::Quaterniond::FromTwoVectors(mPrevAction1.head(m), a_real.head(m)));
	// double action_diff = aa.angle();
	// if(action_diff>0.5){
	// 	double a_real_val = a_real.head(m).norm();
	// 	Eigen::AngleAxisd aa2(0.5,aa.axis());
	// 	a_real.head(m) = aa2.toRotationMatrix()*(mPrevAction1.head(m).normalized());
	// 	a_real.head(m) *= a_real_val;
	// }

	// a_real.tail(n) += Eigen::VectorXd::Ones(n);
	// a_real.tail(n) = Eigen::VectorXd::Ones(n);
	return a_real;
}
Eigen::Vector3d
Environment::
getTargetCenter()
{
	Eigen::Vector3d center = Eigen::Vector3d(-0.0003061, 2.44461, 0.534614);

	double x = mR*std::sin(mTHETA)*std::cos(mPHI);
	double y = mR*std::sin(mTHETA)*std::sin(mPHI);
	double z = mR*std::cos(mTHETA);

	center[0] += x;
	center[1] += y;
	center[2] += z;

	return center;
}
void
Environment::
setRandomTargetCenter()
{
	mR = dart::math::Random::uniform<double>(0.2, 0.4);
	mTHETA = dart::math::Random::uniform<double>(0.0,0.5*M_PI);
	mPHI = dart::math::Random::uniform<double>(0.0,2*M_PI);
}