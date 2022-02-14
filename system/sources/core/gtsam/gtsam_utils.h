#ifndef DF_GTSAM_UTILS_H_
#define DF_GTSAM_UTILS_H_

#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/Key.h>

/*
 * Shorthand functions to get keys for certain keyframe id
 */
inline gtsam::Key AuxPoseKey(uint64_t j) { return gtsam::Symbol('a', j); }
inline gtsam::Key PoseKey(uint64_t j) { return gtsam::Symbol('p', j); }
inline gtsam::Key CodeKey(uint64_t j) { return gtsam::Symbol('c', j); }
inline gtsam::Key ScaleKey(uint64_t j) { return gtsam::Symbol('s', j); }

#endif // DF_GTSAM_UTILS_H_
