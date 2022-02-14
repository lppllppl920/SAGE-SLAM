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

#include <map>
#include <string>
#include <vector>

#pragma once

namespace wrap {

// Forward declarations
class Qualified;
class Class;
struct ForwardDeclaration;

/** Attributes about valid classes, both for classes defined in this module and
 * also those forward-declared from others.  At the moment this only contains
 * whether the class is virtual, which is used to know how to copy the class,
 * and whether to try to convert it to a more derived type upon returning it.
 */
struct TypeAttributes {
  bool isVirtual;
  TypeAttributes() :
      isVirtual(false) {
  }
  TypeAttributes(bool isVirtual) :
      isVirtual(isVirtual) {
  }
};

/** Map of type names to attributes. */
class TypeAttributesTable {

  std::map<std::string, TypeAttributes> table_;

public:

  /// Constructor
  TypeAttributesTable() {
  }

  void addClasses(const std::vector<Class>& classes);
  void addType(const Qualified& types);
  void addForwardDeclarations(
      const std::vector<ForwardDeclaration>& forwardDecls);

  /// Access attributes associated with key, informative failure
  const TypeAttributes& attributes(const std::string& key) const;

  /// Check that all virtual classes are properly defined
  void checkValidity(const std::vector<Class>& classes) const;
};

}
