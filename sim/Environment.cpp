#include <functional>
#include <fstream>
#include <sstream>
#include "Environment.h"
#include "MSD.h"
#include "DARTUtils.h"
#include "MathUtils.h"
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
	:mWorld(std::make_shared<World>()),
	mControlHz(30),
	mSimulationHz(300),
	mElapsedFrame(0),
	mMaxElapsedFrame(300),
	mSimCharacter(nullptr),
	mKinCharacter(nullptr),
	mTargetSpeedMin(0.5),
	mTargetSpeedMax(3.5),
	mSharpTurnProb(0.01),
	mSpeedChangeProb(0.05),
	mMaxHeadingTurnRate(0.15),
	mRewardGoal(0.0),
	mEnableGoal(true)
{
	dart::math::Random::generateSeed(true);

	mSimCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel.xml");
	mKinCharacter = DARTUtils::buildFromFile(std::string(ROOT_DIR)+"/data/skel.xml");
	
	BVH* bvh = new BVH(std::string(ROOT_DIR)+"/data/bvh/walk_long.bvh");
	Motion* motion = new Motion(bvh);
	// for(int j=120;j<bvh->getNumFrames();j++)
		// motion->append(bvh->getPosition(j), bvh->getRotation(j),false);
	for(int i=0;i<120;i++)
		motion->append(bvh->getPosition(85), bvh->getRotation(85),false);
	motion->computeVelocity();
	mMotions.emplace_back(motion);
	mSimCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
	mKinCharacter->buildBVHIndices(motion->getBVH()->getNodeNames());
	this->parseMSD(std::string(ROOT_DIR)+"/data/msd.txt");
	double ground_height = this->computeGroundHeight();
	mGround = DARTUtils::createGround(ground_height);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->addSkeleton(mSimCharacter->getSkeleton());
	mWorld->addSkeleton(mGround);
	mWorld->setTimeStep(1.0/(double)mSimulationHz);
	mWorld->setGravity(Eigen::Vector3d(0,-9.81,0.0));
	mElapsedFrame = 0;
	mFrame = 0;

	mSimCharacter->getSkeleton()->setSelfCollisionCheck(true);
	mSimCharacter->getSkeleton()->setAdjacentBodyCheck(false);

	Eigen::Vector3d k(100.0,100.0,100.0),d(0.6,0.6,0.6),m(1.0,0.0,1.0);
	mForceCartesianMSD = new CartesianMSD(k,d,m, 1.0/(double)mControlHz);
	mForceCartesianMSD->reset();
	mLeftHandTargetProjection.setZero();
	this->reset();

	mActionSpace = this->getActionSpace();
	mActionWeight = this->getActionWeight();
}
void
Environment::
parseMSD(const std::string& file)
{
	auto bvh =  mMotions[0]->getBVH();
	int njoints = mMotions[0]->getNumJoints();
	std::ifstream ifs(file);
	std::string line;
	Eigen::VectorXd m = Eigen::VectorXd::Zero(3+3*njoints);
	Eigen::VectorXd s = Eigen::VectorXd::Zero(3+3*njoints);
	Eigen::VectorXd d = Eigen::VectorXd::Zero(3+3*njoints);
	while(!ifs.eof())
	{
		line.clear();
		std::getline(ifs, line);
		if(line.size() == 0)
			continue;

		std::stringstream ss(line);
		std::string token;
		ss>>token;

		int o = 0;
		if(token == "ROOT")
			o = 0;
		else if(token == "JOINT")
		{
			std::string joint_name;
			ss>>joint_name;
			o = 3+3*bvh->getNodeIndex(joint_name);
		}
		double mi,si,di;
		ss>>mi>>si>>di;

		m.segment<3>(o) = Eigen::Vector3d::Constant(mi);
		s.segment<3>(o) = Eigen::Vector3d::Constant(si);
		d.segment<3>(o) = Eigen::Vector3d::Constant(di);
	}
	// mSimCharacter->buildMSD(s,d,m,bvh->getTimestep());
	mKinCharacter->buildMSD(s,d,m,bvh->getTimestep());
}
int
Environment::
getDimState()
{
	return this->getState().rows();
}
int
Environment::
getDimAction()
{
	int n = mSimCharacter->getSkeleton()->getNumDofs();
	return n-6;
}
int
Environment::
getDimStateAMP()
{
	return this->getStateAMP().rows();
}
void
Environment::
reset(int frame)
{
	mContactEOE = false;
	mFrame = 0;
	mElapsedFrame = 0;

	auto motion = mMotions[0];
	// mFrame = dart::math::Random::uniform<int>(200,motion->getNumFrames()-3);
	mFrame = 0;
	Eigen::Vector3d position = motion->getPosition(mFrame);
	Eigen::MatrixXd rotation = motion->getRotation(mFrame);
	Eigen::Vector3d linear_velocity = motion->getLinearVelocity(mFrame);
	Eigen::MatrixXd angular_velocity = motion->getAngularVelocity(mFrame);

	mKinCharacter->getMSD()->reset();
	mSimCharacter->setPose(position, rotation, linear_velocity, angular_velocity);

	mSimCharacter->getSkeleton()->clearConstraintImpulses();
	mSimCharacter->getSkeleton()->clearInternalForces();
	mSimCharacter->getSkeleton()->clearExternalForces();

	mKinCharacter->setPose(position, rotation, linear_velocity, angular_velocity);

	mForceCartesianMSD->reset();
	mPrevPositions2 = mSimCharacter->getSkeleton()->getPositions(); 
	mPrevPositions = mSimCharacter->getSkeleton()->getPositions();
	mPrevCOM = mSimCharacter->getSkeleton()->getCOM();
	mPrevMSDStates = mKinCharacter->getMSD()->saveState();

	if(mEnableGoal){
		this->resetGoal();
		this->recordGoal();
	}
	this->recordState();
}
void
Environment::
step(const Eigen::VectorXd& _action)
{
	double r = dart::math::Random::uniform<double>(0.0,1.0);
	if(r<0.02)
		mLeftHandTargetProjection = Eigen::Vector3d::Random()*2 - Eigen::Vector3d::Ones();
	mForceCartesianMSD->setProjection(mLeftHandTargetProjection);
	mForceCartesianMSD->step();
	Eigen::Vector3d force = 10.0*mForceCartesianMSD->getPosition();

	mKinCharacter->applyForceMSD("LeftHand", force, Eigen::Vector3d::Zero());

	Eigen::VectorXd action = this->convertToRealActionSpace(_action);

	auto sim_skel = mSimCharacter->getSkeleton();
	int num_sub_steps = mSimulationHz/mControlHz;

	auto target_pos = mSimCharacter->computeTargetPosition(action);

	auto msd = mKinCharacter->getMSD();
	Eigen::MatrixXd tar_rot;
	Eigen::MatrixXd sim_rot = mSimCharacter->getPose();
	Eigen::MatrixXd msd_rot = MotionUtils::addDisplacement(sim_rot, msd->getRotation());
	if(mSimCharacter->getLastForceBodyNodeName().size() != 0)
	{
		std::string force_body_name = sim_skel->getBodyNode(mSimCharacter->getLastForceBodyNodeName())->getName();
		std::string force_sim_body_name = mMotions[0]->getName(mSimCharacter->getBVHIndex(sim_skel->getIndexOf(sim_skel->getBodyNode(force_body_name))));
		tar_rot = MotionUtils::computeWeightedClosestPose(mMotions[0], force_sim_body_name, msd_rot);
	}
	else
		tar_rot = MotionUtils::computeClosestPose(mMotions[0], msd_rot);
	Eigen::Vector3d pos = msd->getPosition();
	pos.setZero();
	pos[1] = 1.0;
	Eigen::MatrixXd rot = MotionUtils::addDisplacement(MotionUtils::TransposeMatrix(sim_rot), tar_rot);
	msd->setProjection(pos, rot);
	mKinCharacter->stepMSD();
	mKinCharacter->setPose(msd->getPosition(), MotionUtils::addDisplacement(sim_rot, msd->getRotation()));

	

	for(int i=0;i<num_sub_steps;i++)
	{
		mSimCharacter->actuate(target_pos);
		mWorld->step();

		auto cr = mWorld->getConstraintSolver()->getLastCollisionResult();

		for(int j=0;j<cr.getNumContacts();j++)
		{
			auto contact = cr.getContact(j);
			auto shapeFrame1 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject1->getShapeFrame());
			auto shapeFrame2 = const_cast<dart::dynamics::ShapeFrame*>(contact.collisionObject2->getShapeFrame());

			auto bn1 = shapeFrame1->asShapeNode()->getBodyNodePtr();
			auto bn2 = shapeFrame2->asShapeNode()->getBodyNodePtr();

			auto skel1 = bn1->getSkeleton();
			auto skel2 = bn2->getSkeleton();

			if(bn1->getName().find("Foot") != std::string::npos)
				continue;
			else if(bn2->getName().find("Foot") != std::string::npos)
				continue;

			if(skel1->getName() == "humanoid" && skel2->getName() == "ground"){
				mContactEOE = true;
				break;
			}

			if(skel1->getName() == "ground" && skel2->getName() == "humanoid"){
				mContactEOE = true;
				break;
			}

		}
	}
	if(mEnableGoal)
	{
		this->recordGoal();
		this->updateGoal();
	}
	
	this->recordState();

	mPrevPositions2 = mPrevPositions;
	mPrevPositions = mSimCharacter->getSkeleton()->getPositions();
	mPrevCOM = mSimCharacter->getSkeleton()->getCOM();
	mPrevMSDStates = mKinCharacter->getMSD()->saveState();

	mElapsedFrame++;
	mFrame++;
}
void
Environment::
resetGoal()
{
	return;
}
void
Environment::
updateGoal()
{
	return;
}
void
Environment::
recordGoal()
{
	mRewardGoal = 1.0;
}

double
Environment::
getRewardGoal()
{
	return mRewardGoal;
}


const Eigen::VectorXd&
Environment::
getStateGoal()
{
	return mStateGoal;
}

const Eigen::VectorXd&
Environment::
getState()
{
	return mState;
}
const Eigen::VectorXd&
Environment::
getStateAMP()
{
	return mStateAMP;
}
void
Environment::
recordState()
{
	Eigen::VectorXd state = MathUtils::ravel(mSimCharacter->getState());
	if(mEnableGoal)
	{
		Eigen::VectorXd goal = this->getStateGoal();
		mState = Eigen::VectorXd(state.rows() + goal.rows());
		mState<<state, goal;	
	}
	else
		mState = state;	
	
	auto save_state = mKinCharacter->saveState();
	auto save_msd_state = mKinCharacter->getMSD()->saveState();

	Eigen::VectorXd prev_velocities = mSimCharacter->computeAvgVelocity(mPrevPositions2, mPrevPositions, 1.0/mControlHz);
	mKinCharacter->getSkeleton()->setPositions(mPrevPositions);
	mKinCharacter->getSkeleton()->setVelocities(prev_velocities);
	mKinCharacter->getMSD()->restoreState(mPrevMSDStates);

	Eigen::VectorXd s = mKinCharacter->getStateAMP();

	Eigen::VectorXd velocities = mSimCharacter->computeAvgVelocity(mPrevPositions, mSimCharacter->getSkeleton()->getPositions(), 1.0/mControlHz);
	mKinCharacter->getSkeleton()->setPositions(mSimCharacter->getSkeleton()->getPositions());
	mKinCharacter->getSkeleton()->setVelocities(velocities);
	mKinCharacter->getMSD()->restoreState(save_msd_state);

	Eigen::VectorXd s1 = mKinCharacter->getStateAMP();
	mKinCharacter->restoreState(save_state);
	mStateAMP.resize(s.rows() + s1.rows());
	mStateAMP<<s, s1;
}


Eigen::MatrixXd
Environment::
getStateAMPExpert()
{
	int total_num_frames = 0;
	int m = this->getDimStateAMP();
	int m2 = m/2;
	int o = 0;
	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		total_num_frames += nf-1;
	}
	Eigen::MatrixXd state_expert(total_num_frames,m);

	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		mKinCharacter->setPose(motion->getPosition(0),
							motion->getRotation(0),
							motion->getLinearVelocity(0),
							motion->getAngularVelocity(0));
		Eigen::VectorXd s = mKinCharacter->getStateAMP();

		Eigen::VectorXd s1;

		for(int i=0;i<nf-1;i++)
		{
			mKinCharacter->setPose(motion->getPosition(i+1),
							motion->getRotation(i+1),
							motion->getLinearVelocity(i+1),
							motion->getAngularVelocity(i+1));
			s1 = mKinCharacter->getStateAMP();

			state_expert.row(o+i).head(m2) = s.transpose();
			state_expert.row(o+i).tail(m2) = s1.transpose();
			s = s1;
		}
		o += nf - 1;
	}
	return state_expert;
}
bool
Environment::
inspectEndOfEpisode()
{
	if(mContactEOE)
		return true;
	else if(mElapsedFrame>mMaxElapsedFrame)
		return true;

	return false;
}

double
Environment::
computeGroundHeight()
{
	double y = 1e6;

	for(auto motion: mMotions)
	{
		int nf = motion->getNumFrames();
		for(int i=0;i<nf;i++)
		{
			mKinCharacter->setPose(motion->getPosition(i),
							motion->getRotation(i),
							motion->getLinearVelocity(i),
							motion->getAngularVelocity(i));

			y= std::min(y, mKinCharacter->getSkeleton()->getBodyNode("LeftFoot")->getCOM()[1]);
			y= std::min(y, mKinCharacter->getSkeleton()->getBodyNode("RightFoot")->getCOM()[1]);
		}
	}
	
	float dy = dynamic_cast<const BoxShape*>(mKinCharacter->getSkeleton()->getBodyNode("LeftFoot")->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	return y - dy;
}
Eigen::MatrixXd
Environment::
getActionSpace()
{
	Eigen::MatrixXd action_space = Eigen::MatrixXd::Ones(this->getDimAction(), 2);
	int n = mSimCharacter->getSkeleton()->getNumDofs();

	action_space.col(0) *= -M_PI*2; // Lower
	action_space.col(1) *=  M_PI*2; // Upper

	return action_space;
}
Eigen::VectorXd
Environment::
getActionWeight()
{
	Eigen::VectorXd action_weight = Eigen::VectorXd::Ones(this->getDimAction());
	int n = mSimCharacter->getSkeleton()->getNumDofs();

	return action_weight;
}
Eigen::VectorXd
Environment::
convertToRealActionSpace(const Eigen::VectorXd& a_norm)
{
	Eigen::VectorXd a_real;
	Eigen::VectorXd lo = mActionSpace.col(0), hi =  mActionSpace.col(1);
	a_real = dart::math::clip<Eigen::VectorXd, Eigen::VectorXd>(a_norm, lo, hi);
	a_real = mActionWeight.cwiseProduct(a_real);
	return a_real;
}