#include "Motion.h"
#include "BVH.h"
#include <Eigen/Geometry>
#include <iostream>

Motion::
Motion(BVH* bvh)
{
	this->registerBVHHierarchy(bvh);
	// this->append(bvh->getPositions(), bvh->getRotations(),true);
}
void
Motion::
append(const Eigen::Vector3d& position,
		const Eigen::MatrixXd& rotation,
		bool compute_velocity)
{
	mPositions.emplace_back(position);
	mRotations.emplace_back(rotation);
	
	mNumFrames = mPositions.size();

	if(compute_velocity)
	{
		int n = mNumFrames;
		int m = mLinearVelocities.size();
		m = std::max(0,m-1); // compute last velocity again
		// std::cout<<m<<" "<<n<<std::endl;
		this->computeVelocity(m,n);
	}
}
void
Motion::
append(const std::vector<Eigen::Vector3d>& positions,
		const std::vector<Eigen::MatrixXd>& rotations,
		bool compute_velocity)
{
	mPositions.insert(mPositions.end(), positions.begin(), positions.end());
	mRotations.insert(mRotations.end(), rotations.begin(), rotations.end());
	mNumFrames = mPositions.size();

	if(compute_velocity)
	{
		int n = mNumFrames;
		int m = mLinearVelocities.size();
		m = std::max(0,m-1); // compute last velocity again
		this->computeVelocity(m,n);
	}
}

void
Motion::
clear()
{
	mPositions.clear();
	mRotations.clear();
	mLinearVelocities.clear();
	mAngularVelocities.clear();

	mNumFrames = 0;
}
void
Motion::
registerBVHHierarchy(BVH* bvh)
{
	mBVH = bvh;
	mTimestep = bvh->getTimestep();
	mNames = bvh->getNodeNames();
	mOffsets = bvh->getOffsets();
	mParents = bvh->getParents();
	mNumJoints = mParents.size();
}
void
Motion::
computeVelocity()
{
	this->computeVelocity(0);
}
void
Motion::
computeVelocity(int start)
{
	this->computeVelocity(start,mNumFrames);
}
void
Motion::
computeVelocity(int start, int end)
{
	assert(end <= mNumFrames);
	int n = end - start;
	std::vector<Eigen::Vector3d> linear_velocities(n);
	std::vector<Eigen::MatrixXd> angular_velocities(n);

	for(int i=0;i<n;i++)
	{
		int idx = start + i;
		int frame1 = std::max(0,idx-1);
		int frame2 = std::min(mNumFrames-1,idx+1);

		double dt_inv = 1.0/(mTimestep*(frame2-frame1));
		if(frame1==frame2)
			dt_inv = 0.0;

		Eigen::Vector3d pos1 = mPositions[frame1];
		Eigen::MatrixXd rot1 = mRotations[frame1];

		Eigen::Vector3d pos2 = mPositions[frame2];
		Eigen::MatrixXd rot2 = mRotations[frame2];

		linear_velocities[i] = (pos2 - pos1)*dt_inv;

		Eigen::Matrix3d R1 = rot1.block<3,3>(0,0);
		Eigen::Matrix3d R2 = rot2.block<3,3>(0,0);

		Eigen::Quaterniond Q1(R1);
		Eigen::Quaterniond Q2(R2);
		Eigen::Vector4d q1;
		Eigen::Vector4d q2;
		q1<<Q1.vec(),Q1.w();
		q2<<Q2.vec(),Q2.w();
		if( (q1-q2).norm() < (q1+q2).norm())
			Q2 = Eigen::Quaterniond(q2[3],q2[0],q2[1],q2[2]);

		Eigen::AngleAxisd aa(Q2 * Q1.conjugate()); // Not sure
		angular_velocities[i].resize(3,mNumJoints);
		
		angular_velocities[i].col(0) = aa.angle()*aa.axis();
		for(int j=1;j<mNumJoints;j++)
		{
			Q1 = Eigen::Quaterniond(rot1.block<3,3>(0,j*3));
			Q2 = Eigen::Quaterniond(rot2.block<3,3>(0,j*3));
			aa = Eigen::AngleAxisd(Q1.conjugate()*Q2);
			angular_velocities[i].col(j) = aa.angle()*aa.axis();
		}
		angular_velocities[i] *= dt_inv;
	}

	if(mLinearVelocities.size()>=start)
	{
		mLinearVelocities.erase(mLinearVelocities.begin()+start,mLinearVelocities.end());
		mAngularVelocities.erase(mAngularVelocities.begin()+start,mAngularVelocities.end());
	}
	// assert(mLinearVelocities.size() == start-1 || mLinearVelocities.size() == 0);
	mLinearVelocities.insert(mLinearVelocities.end(),linear_velocities.begin(),linear_velocities.end());
	mAngularVelocities.insert(mAngularVelocities.end(),angular_velocities.begin(),angular_velocities.end());

	// std::cout<<mLinearVelocities.size()<<std::endl;
	// std::cout<<mPositions.size()<<std::endl;
}








Eigen::MatrixXd
MotionUtils::
computePoseDifferences(Motion* m)
{
	int n = m->getNumFrames();
	Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n, n);
	for(int i=0;i<n;i++)
	{
		const Eigen::MatrixXd& Ri = m->getRotation(i);
		for(int j=i+1;j<n;j++)
		{
			const Eigen::MatrixXd& Rj = m->getRotation(j);
			double dij = computePoseDifference(Ri, Rj);
			D(i, j) = dij;
			D(j, i) = dij;
		}
	}
	return D;
}

double
MotionUtils::
computePoseDifference(const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rj)
{
	int n = Ri.cols()/3;
	double d = 0.0;
	for(int i=0;i<n;i++)
	{
		Eigen::AngleAxisd aa(Ri.block<3,3>(0,i*3).transpose()*Rj.block<3,3>(0,i*3));
		//ToDO : add weights
		d += aa.angle();
	}

	return d;
}
Eigen::MatrixXd
MotionUtils::
computePoseDisplacement(const Eigen::MatrixXd& Ri, const Eigen::MatrixXd& Rj)
{
	int n = Ri.cols()/3;
	Eigen::MatrixXd d(4,n);
	for(int i=0;i<n;i++)
	{
		Eigen::Quaterniond q(Ri.block<3,3>(0,i*3).transpose()*Rj.block<3,3>(0,i*3));
		d(0,i) = q.w();
		d.block<3,1>(1,i) = q.vec();
	}

	return d;
}
Eigen::MatrixXd
MotionUtils::
addDisplacement(const Eigen::MatrixXd& R, const Eigen::MatrixXd& d)
{
	int n = R.cols()/3;
	Eigen::MatrixXd Rd = R;
	for(int i=0;i<n;i++)
	{
		Eigen::Vector4d d_i = d.col(i);
		if(d_i[1]>1.0-1e-6)
			continue;

		Eigen::Quaterniond q(d_i[0],d_i[1],d_i[2],d_i[3]);
		Rd.block<3,3>(0,i*3) = R.block<3,3>(0,i*3)*q.toRotationMatrix();
	}

	return Rd;
}
double
MotionUtils::
easeInEaseOut(double x, double yp0, double yp1)
{
	double y = (x-1.0)*((yp0+yp1+2)*x*x - (yp0 + 1)*x - 1.0);
	// double y = 2*x*x*x - 3*x*x + 1.0;
	return y;
	// return std::max(0.0,std::min(1.0,y));
}