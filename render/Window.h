#ifndef __WINDOW_H__
#define __WINDOW_H__
#include "GLUTWindow3D.h"
#include <utility>
#include <chrono>
#include <Eigen/Core>
#include "Environment.h"
#include "DARTRendering.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
namespace py = pybind11;

class Window : public GLUTWindow3D
{
public:
	Window();

	Environment* mEnvironment;
	void render() override;

	void reset(int frame=-1);
	void step();

	void initNN(const std::string& config);
	void loadNN(const std::string& checkpoint);
protected:
	void keyboard(unsigned char key, int x, int y) override;
	void special(int key, int x, int y) override;
	void mouse(int button, int state, int x, int y) override;
	void motion(int x, int y) override;
	void reshape(int w, int h) override;
	void timer(int tic) override;

	std::vector<unsigned char> mScreenshotTemp;
	std::vector<unsigned char> mScreenshotTemp2;
	bool mCapture;
	void capture_screen();

	bool mPlay;
	
	DARTRendering::Option mSimRenderOption;
	DARTRendering::Option mKinRenderOption;
	DARTRendering::Option mTargetRenderOption;
	DARTRendering::Option mObjectRenderOption;

	bool mUseNN;

	bool mDrawSimPose, mDrawKinPose, mDrawTargetPose, mDrawCOMvel, mDraw2DCharacter;
	bool mExplore, mFocus;
	double mReward, mRewardGoal;
	bool mPlotReward;

	Eigen::VectorXd mObservation, mObservationDiscriminator;
	DrawUtils::BarPlot mBarPlot;
	py::scoped_interpreter guard;
	std::vector<double> mRewards, mRewardGoals;
	py::object mm,mns,sys_module;
	py::module policy_md, discriminator_md;
	py::object policy, discriminator;

	std::chrono::system_clock::time_point mTimePoint;
	double mComputedTime;
};

#endif