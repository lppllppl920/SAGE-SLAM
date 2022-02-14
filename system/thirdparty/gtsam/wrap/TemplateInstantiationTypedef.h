/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Class.h
 * @brief describe the C++ class that is being wrapped
 * @author Frank Dellaert
 * @author Andrew Melim
 * @author Richard Roberts
 **/

#pragma once

#include <vector>
#include <string>

#include "Class.h"

namespace wrap {

struct TemplateInstantiationTypedef : public Qualified {
  Qualified class_;
  std::vector<Qualified> typeList;

  Class findAndExpand(const std::vector<Class>& classes) const;
};

}
