#include "interp.h"

Eigen::Quaternionf QuatDamp(const Eigen::Quaternionf& from,
                            const Eigen::Quaternionf& to,
                            Eigen::Vector4f& vel,
                            float delta_time,
                            float damping,
                            float max_speed)
{
  auto dot = from.dot(to);
  if (dot >= 0.9999)
    return to;

  // Use the closest rotation
  Eigen::Quaternionf q_from = dot < 0 ? Eigen::Quaternionf(-from.coeffs()) : from;

  // n1 = velocity - ( currentRot - targerRot  ) * ( omega * omega * dt );
  // n2 = 1 + omega * dt;
  // newVelocity = n1 / ( n2 * n2 );
  // currentRot += newVelocity * dt;
  float dSqrDt = damping * damping * delta_time;
  float n2 = 1 + damping * delta_time;
  float n2Sqr	= n2 * n2;
  vel = (vel - (q_from.coeffs() - to.coeffs()) * dSqrDt) / n2Sqr;

  //.........................................
  Eigen::Quaternionf q_new(q_from);
  q_new.coeffs() += vel * delta_time;
  q_new.normalize();
  return q_new;
}

Sophus::SE3f Interpolate(float x, const Sophus::SE3f& pose0, const Sophus::SE3f& pose1)
{
  Eigen::Vector3f trs = lerp(x, pose0.translation(), pose1.translation());
  Eigen::Quaternionf rot = pose0.so3().unit_quaternion().slerp(x, pose1.so3().unit_quaternion());
  return Sophus::SE3f(rot, trs);
}
