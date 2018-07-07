/**
 * @file ProgramOptionsHelper.h
 * @brief ProgramOptionsHelper
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_UTILS_PROGRAM_OPTIONS_HELPER_H_
#define VIDEOPARSING_UTILS_PROGRAM_OPTIONS_HELPER_H_

#include <boost/program_options.hpp>

namespace vp {

// makes dataset, start_frame and number_of_frames
boost::program_options::options_description makeDatasetConfigOptions();

boost::program_options::options_description makeKernelParamsOptions();
void verfyKernelParamsOptions(const boost::program_options::variables_map& vm);


boost::program_options::options_description makeHOSegementPotentialOptions();
boost::program_options::options_description makeHOTrackPotentialOptions();


void printPOVariablesMap(const boost::program_options::variables_map& vm);


}  // namespace vp

#endif /* VIDEOPARSING_UTILS_PROGRAM_OPTIONS_HELPER_H_ */
