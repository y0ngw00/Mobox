// #include <pybind11/embed.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/eigen.h>
// #include <pybind11/stl.h>
// #include "Environment.h"
// namespace py = pybind11;

// PYBIND11_MODULE(pycomcon, m){
// 	py::class_<Environment>(m, "env")
// 		.def(py::init<>())
// 		.def("reset",&Environment::reset)
// 		.def("step",&Environment::step)
// 		.def("get_state",&Environment::getState)
// 		.def("get_reward",&Environment::getReward)
// 		.def("get_info",&Environment::getInfo)
// 		.def("inspect_end_of_episode",&Environment::inspectEndOfEpisode)
// 		.def("get_dim_state",&Environment::getDimState)
// 		.def("get_dim_action",&Environment::getDimAction)
// 		.def("get_average_force_reward",&Environment::getAverageForceReward)
// 		.def("get_force_distribution",&Environment::getForceDistribution)
// 		.def("set_force_distribution",&Environment::setForceDistribution)
// 		.def("get_ball_distribution",&Environment::getBallDistribution)
// 		.def("set_ball_distribution",&Environment::setBallDistribution)
// 		.def("set_current_force_boundary",&Environment::setCurrentForceBoundary);
// }