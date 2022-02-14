/**
 * @file ReturnValue.cpp
 * @date Dec 1, 2011
 * @author Alex Cunningham
 * @author Andrew Melim
 * @author Richard Roberts
 */

#include "ReturnValue.h"
#include "utilities.h"
#include <iostream>

using namespace std;
using namespace wrap;

/* ************************************************************************* */
ReturnValue ReturnValue::expandTemplate(const TemplateSubstitution& ts) const {
  ReturnValue instRetVal = *this;
  instRetVal.type1 = ts.tryToSubstitite(type1);
  if (isPair) instRetVal.type2 = ts.tryToSubstitite(type2);
  return instRetVal;
}

/* ************************************************************************* */
string ReturnValue::returnType() const {
  if (isPair)
    return "pair< " + type1.qualifiedName("::") + ", " +
           type2.qualifiedName("::") + " >";
  else
    return type1.qualifiedName("::");
}

/* ************************************************************************* */
string ReturnValue::matlab_returnType() const {
  return isPair ? "[first,second]" : "result";
}

/* ************************************************************************* */
void ReturnValue::wrap_result(const string& result, FileWriter& wrapperFile,
                              const TypeAttributesTable& typeAttributes) const {
  if (isPair) {
    // For a pair, store the returned pair so we do not evaluate the function
    // twice
    wrapperFile.oss << "  auto pairResult = " << result
                    << ";\n";
    type1.wrap_result("  out[0]", "pairResult.first", wrapperFile,
                      typeAttributes);
    type2.wrap_result("  out[1]", "pairResult.second", wrapperFile,
                      typeAttributes);
  } else {  // Not a pair
    type1.wrap_result("  out[0]", result, wrapperFile, typeAttributes);
  }
}

/* ************************************************************************* */
void ReturnValue::emit_matlab(FileWriter& proxyFile) const {
  string output;
  if (isPair)
    proxyFile.oss << "[ varargout{1} varargout{2} ] = ";
  else if (type1.category != ReturnType::VOID)
    proxyFile.oss << "varargout{1} = ";
}

/* ************************************************************************* */
void ReturnValue::emit_cython_pxd(
    FileWriter& file, const std::string& className,
    const std::vector<std::string>& templateArgs) const {
  if (isPair) {
    file.oss << "pair[";
    type1.emit_cython_pxd(file, className, templateArgs);
    file.oss << ",";
    type2.emit_cython_pxd(file, className, templateArgs);
    file.oss << "] ";
  } else {
    type1.emit_cython_pxd(file, className, templateArgs);
    file.oss << " ";
  }
}

/* ************************************************************************* */
std::string ReturnValue::pyx_returnType() const {
  if (isVoid()) return "";
  if (isPair) {
    return "pair [" + type1.pyx_returnType(false) + "," +
           type2.pyx_returnType(false) + "]";
  } else {
    return type1.pyx_returnType(true);
  }
}

/* ************************************************************************* */
std::string ReturnValue::pyx_casting(const std::string& var) const {
  if (isVoid()) return "";
  if (isPair) {
    return "(" + type1.pyx_casting(var + ".first", false) + "," +
           type2.pyx_casting(var + ".second", false) + ")";
  } else {
    return type1.pyx_casting(var);
  }
}

/* ************************************************************************* */
