#include "Character.h"
#include "Environment.h"
#include "DARTUtils.h"
using namespace dart;
using namespace dart::dynamics;

Event::
Event(Environment* env, Character* character)
	:mEnvironment(env), mCharacter(character)
{

}


ObstacleEvent::
ObstacleEvent(Environment* env, Character* character)
	:Event(env, character),mTriggerCount(0),mTriggerDuration(0)
{
	// if(0)
	{
		mObstacle = DARTUtils::createBall(1e9, 0.03, "Free");
		mEnvironment->getWorld()->addSkeleton(mObstacle);
		mObstacle->getJoint(0)->setDampingCoefficient(0,0.1);
		mObstacle->getJoint(0)->setDampingCoefficient(1,0.1);
		mObstacle->getJoint(0)->setDampingCoefficient(2,0.1);
		mObstacle->getJoint(0)->setDampingCoefficient(3,0.003);
		mObstacle->getJoint(0)->setDampingCoefficient(4,0.003);
		mObstacle->getJoint(0)->setDampingCoefficient(5,0.003);	
	}
	
	// mObstacle = DARTUtils::createBall(4000000.0, 0.15, "Weld");
	// mEnvironment->getWorld()->addSkeleton(mObstacle);
	// Eigen::Isometry3d T_ref;
	// T_ref.linear() = Eigen::Matrix3d::Identity();
	// T_ref.translation() = Eigen::Vector3d::UnitY()*3.0;
	// mObstacle->getJoint(0)->setTransformFromParentBodyNode(T_ref);
	// mObstacle->setMobile(false);
	// mLinearVelocity.setZero();
	mLinearPosition.setZero();
	mLinearVelocity.setZero();
}
void
ObstacleEvent::
call()
{
	return;
	// Eigen::Isometry3d T_c = mObstacle->getJoint(0)->getTransformFromParentBodyNode();
	// T_c.translation() += mLinearVelocity/30.0;
	// mObstacle->getJoint(0)->setTransformFromParentBodyNode(T_c);
	// Eigen::Vector3d minb(-0.2,,0.2);

	Eigen::Isometry3d T_bn = mCharacter->getSkeleton()->getBodyNode(0)->getShapeNodesWith<VisualAspect>().back()->getTransform();
	T_bn.linear() = Eigen::Matrix3d::Identity();
	auto bb = mCharacter->getSkeleton()->getBodyNode(0)->getShapeNodesWith<VisualAspect>().back()->getShape().get()->getBoundingBox();
	Eigen::Vector3d p1 = T_bn*bb.computeCenter();
	T_bn = mCharacter->getSkeleton()->getBodyNode(1)->getShapeNodesWith<VisualAspect>().back()->getTransform();
	T_bn.linear() = Eigen::Matrix3d::Identity();
	bb = mCharacter->getSkeleton()->getBodyNode(1)->getShapeNodesWith<VisualAspect>().back()->getShape().get()->getBoundingBox();
	Eigen::Vector3d p2 = T_bn*bb.computeCenter();

	bb = dart::math::BoundingBox(p1.cwiseMin(p2)-Eigen::Vector3d(0.08,0.17, 0.08), p1.cwiseMax(p2)+Eigen::Vector3d(0.08,0.1,0.08));

	Eigen::Vector3d o_pos = mLinearPosition;
	double o_rad = dynamic_cast<SphereShape*>(mObstacle->getBodyNode(0)->getShapeNodesWith<VisualAspect>().back()->getShape().get())->getRadius();

	// get box closest point to sphere center by clamping
	Eigen::Vector3d clipedp = bb.getMin().cwiseMax(o_pos.cwiseMin(bb.getMax()));

	double distance = (clipedp - o_pos).norm();
	if(distance<o_rad){
		mLinearVelocity.setZero();
		// mLinearPosition.setZero();
	}
	mLinearPosition += mLinearVelocity/30.0;
	// DrawUtils::drawBox(bb.computeCenter(), bb.computeFullExtents());


	mTriggerCount++;
	if(mTriggerCount<mTriggerDuration)
		return;
	// mTriggerDuration = dart::math::Random::uniform<int>(60,90);
	mTriggerDuration = 100;
	mTriggerCount = 0;

	Eigen::Isometry3d T_ref = mCharacter->getReferenceTransform();
	Eigen::Isometry3d T_offset;
	T_offset.linear() = Eigen::Matrix3d::Identity();
	T_offset.translation() = Eigen::AngleAxisd(dart::math::Random::uniform<double>(-M_PI, M_PI), Eigen::Vector3d::UnitY()).toRotationMatrix()*Eigen::Vector3d(0.0,2.2,0.3);
	T_offset.translation()[1] += dart::math::Random::uniform<double>(-0.3, 1.0);
	// T_offset.translation() = Eigen::Vector3d(0.22,2.2,0.5);
	T_ref = T_ref*T_offset;

	double dt = 1.0;//(mTriggerDuration-30.0)/30.0;

	// Eigen::Vector3d target = mCharacter->getSkeleton()->getCOM();
	std::vector<int> random_indices = {2, 3, 4, 5, 6 , 7, 8};

	Eigen::Vector3d target = mCharacter->getSkeleton()->getBodyNode(dart::math::Random::uniform<int>(0,7))->getCOM();	
	mLinearPosition = T_ref.translation();
	mLinearVelocity = (target - T_ref.translation())/dt;
	// mLinearPosition = Eigen::Vector3d(-0.138683, 2.39646, -0.640382);
	// mLinearVelocity = Eigen::Vector3d(0.0512511, -0.090334,-0.268408);

	
	// target[0] += dart::math::Random::uniform<double>(-0.2, 0.2);
	// target[1] += dart::math::Random::uniform<double>(0.5, 0.6);
	// target[2] += dart::math::Random::uniform<double>(-0.4, 0.4);
	// T_ref.translation()[1] += dart::math::Random::uniform<double>(0.8, 1.2);
	// T_ref.translation()[0] += 0.35;
	// T_ref.translation()[1] += 0.8;
	// std::cout<<T_ref.translation().transpose()<<std::endl;
	// std::cout<<target.transpose()<<std::endl;
	
	
	// mObstacle->getJoint(0)->setTransformFromParentBodyNode(T_ref);
	// mObstacle->computeForwardKinematics(	);
	// mObstacle->computeForwardDynamics();

	if(0)
	{
		mTriggerCount++;
		if(mTriggerCount<mTriggerDuration)
			return;
		mTriggerDuration = dart::math::Random::uniform<int>(15,60);
		mTriggerCount = 0;
	
		Eigen::Isometry3d T_ref = mCharacter->getReferenceTransform();
		Eigen::Isometry3d T_offset;
		T_offset.linear() = Eigen::Matrix3d::Identity();
		T_offset.translation() = Eigen::AngleAxisd(dart::math::Random::uniform<double>(-M_PI, M_PI), Eigen::Vector3d::UnitY()).toRotationMatrix()*Eigen::Vector3d(0.0,1.5,3.0);
		T_ref = T_ref*T_offset;

		double dt = 0.4;

		Eigen::Vector3d target = mCharacter->getSkeleton()->getCOM() + dt*mCharacter->getSkeleton()->getCOMLinearVelocity();
		target[0] += dart::math::Random::uniform<double>(-0.2, 0.2);
		target[1] += dart::math::Random::uniform<double>(0.2, 0.6);
		
		Eigen::Vector3d lin_vel = (target - T_ref.translation())/dt;
		lin_vel[1] = 9.81*dt;
		T_ref.translation()[1] = target[1] - 0.5*9.81*dt*dt;

		mObstacle->clearInternalForces();
		mObstacle->clearExternalForces();
		mObstacle->setPositions(Eigen::compose(Eigen::Vector3d::Zero(),T_ref.translation()));
		mObstacle->setVelocities(Eigen::compose(Eigen::Vector3d::Zero(),lin_vel));
	}



	// Eigen::Vector3d target = mCharacter->getSkeleton()->getBodyNode("LeftHand")->getCOM();
	// Eigen::Vector3d start = target;
	// start[1] += 0.5;
	// mObstacle->clearInternalForces();
	// mObstacle->clearExternalForces();
	// mObstacle->setPositions(Eigen::compose(Eigen::Vector3d::Zero(),start));
	// mObstacle->setVelocities(Eigen::compose(Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero()));
}
void
ObstacleEvent::
reset()
{
	Eigen::VectorXd pos = Eigen::VectorXd::Zero(mObstacle->getNumDofs());
	Eigen::VectorXd vel = Eigen::VectorXd::Zero(mObstacle->getNumDofs());
	mObstacle->setPositions(pos);
	mObstacle->setVelocities(vel);
	// Eigen::Isometry3d T_ref;
	// T_ref.linear() = Eigen::Matrix3d::Identity();
	// T_ref.translation() = Eigen::Vector3d::UnitY()*3.0;
	// mObstacle->getJoint(0)->setTransformFromParentBodyNode(T_ref);
	mLinearVelocity.setZero();
	mLinearPosition.setZero();

	mTriggerCount = 0;
	mTriggerDuration = 0;
}
