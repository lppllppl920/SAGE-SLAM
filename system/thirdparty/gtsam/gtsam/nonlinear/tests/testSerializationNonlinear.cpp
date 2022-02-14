/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testSerializationNonlinear.cpp
 * @brief
 * @author Richard Roberts
 * @date Feb 7, 2012
 */

#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3Bundler.h>

#include <gtsam/base/serializationTestHelpers.h>
#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;
using namespace gtsam::serializationTestHelpers;

/* ************************************************************************* */
// Export all classes derived from Value
GTSAM_VALUE_EXPORT(gtsam::Cal3_S2);
GTSAM_VALUE_EXPORT(gtsam::Cal3Bundler);
GTSAM_VALUE_EXPORT(gtsam::Point3);
GTSAM_VALUE_EXPORT(gtsam::Pose3);
GTSAM_VALUE_EXPORT(gtsam::Rot3);
GTSAM_VALUE_EXPORT(gtsam::PinholeCamera<Cal3_S2>);
GTSAM_VALUE_EXPORT(gtsam::PinholeCamera<Cal3DS2>);
GTSAM_VALUE_EXPORT(gtsam::PinholeCamera<Cal3Bundler>);

namespace detail {
template<class T> struct pack {
 typedef T type;
};
}

/* ************************************************************************* */
typedef PinholeCamera<Cal3_S2>        PinholeCal3S2;
typedef PinholeCamera<Cal3DS2>        PinholeCal3DS2;
typedef PinholeCamera<Cal3Bundler>    PinholeCal3Bundler;

/* ************************************************************************* */
static Point3 pt3(1.0, 2.0, 3.0);
static Rot3 rt3 = Rot3::RzRyRx(1.0, 3.0, 2.0);
static Pose3 pose3(rt3, pt3);

static Cal3_S2 cal1(1.0, 2.0, 0.3, 0.1, 0.5);
static Cal3DS2 cal2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
static Cal3Bundler cal3(1.0, 2.0, 3.0);

TEST (Serialization, TemplatedValues) {
  EXPECT(equalsObj(pt3));
  GenericValue<Point3> chv1(pt3);
  EXPECT(equalsObj(chv1));
  PinholeCal3S2 pc(pose3,cal1);
  EXPECT(equalsObj(pc));

  Values values;
  values.insert(1,pt3);

  EXPECT(equalsObj(values));
  values.insert(Symbol('a',0),  PinholeCal3S2(pose3, cal1));
  values.insert(Symbol('s',5), PinholeCal3DS2(pose3, cal2));
  values.insert(Symbol('d',47), PinholeCal3Bundler(pose3, cal3));
  values.insert(Symbol('a',5),  PinholeCal3S2(pose3, cal1));
  EXPECT(equalsObj(values));
  EXPECT(equalsXML(values));
  EXPECT(equalsBinary(values));
}

/* ************************************************************************* */
int main() { TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
