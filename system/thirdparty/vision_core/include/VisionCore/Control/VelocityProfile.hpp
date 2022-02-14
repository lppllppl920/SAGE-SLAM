/**
 * ****************************************************************************
 * Copyright (c) 2015, Robert Lukierski.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * ****************************************************************************
 * Velocity Profile Generators.
 * ****************************************************************************
 */

#ifndef VISIONCORE_CONTROL_VELOCITY_PROFILE_HPP
#define VISIONCORE_CONTROL_VELOCITY_PROFILE_HPP

#include <VisionCore/Platform.hpp>

namespace vc
{
    
namespace control
{

/**
 * Velocity Profile Base Methods.
 */
template<typename PT, typename TT = PT, typename VT = PT, typename AT = PT>
class VelocityProfileBase
{
public:
    typedef PT PositionType;
    typedef TT TimeType;
    typedef VT VelocityType;
    typedef AT AccelerationType;
    
    inline VelocityProfileBase()
    {
        max_acc = AccelerationType(0.0);
        max_vel = VelocityType(0.0);
        setup();
    }
    
    inline VelocityProfileBase(VelocityType avel_max, AccelerationType aacc_max) : max_vel(avel_max), max_acc(aacc_max)
    {
        setup();
    }

    inline bool getFinished() const { return is_finished; }
    inline void setMaxVelocity(VelocityType avel_max) { max_vel = avel_max; }
    inline void setMaxAcceleration(AccelerationType aacc_max) { max_acc = aacc_max; }

    inline void reset()
    {
        pos = prev_pos = PositionType(0.0);
        vel = prev_vel = VelocityType(0.0);
        acc = AccelerationType(0.0);
    }
    
    inline PositionType position() const { return pos; }
    inline VelocityType velocity() const { return vel; }
    inline AccelerationType acceleration() const { return acc; }
    
protected:
    void setup()
    {
        is_finished = false;
        reset();
    }
    
    inline void calcState(TimeType delta)
    {
        vel = (pos - prev_pos) / delta;
        if(vel != vel)  // NaN check
        {
            vel = VelocityType(0.0);
        }
        
        // Calculate acceleration
        acc = (vel - prev_vel) / delta;
        if(acc != acc) // NaN check
        {
            acc = AccelerationType(0.0);
        }
    }
    
    VelocityType max_vel;
    AccelerationType max_acc;

    PositionType pos;
    PositionType prev_pos;
    VelocityType vel;
    VelocityType prev_vel;
    AccelerationType acc;
    
    bool is_finished;
};
    
/**
 * Trapezoidal Velocity Profile.
 */
template<typename PT, typename TT = PT, typename VT = PT, typename AT = PT>
class TrapezoidalVelocityProfile : public VelocityProfileBase<PT,TT,VT,AT>
{
    typedef VelocityProfileBase<PT,TT,VT,AT> BaseType;
public:
    typedef PT PositionType;
    typedef TT TimeType;
    typedef VT VelocityType;
    typedef AT AccelerationType;
    
    inline TrapezoidalVelocityProfile() : BaseType()
    {
        
    }
    
    inline TrapezoidalVelocityProfile(VelocityType avel_max, AccelerationType aacc_max) : BaseType(avel_max, aacc_max)
    {
        
    }
    
    inline PositionType operator()(PositionType setPoint, TimeType dt, PositionType eps = std::numeric_limits<PositionType>::epsilon())
    {
        BaseType::is_finished = ( fabs(setPoint - BaseType::pos) < eps );
        
        if(!BaseType::is_finished)
        {
            BaseType::prev_pos = BaseType::pos;
            BaseType::prev_vel = BaseType::vel;
            
            // Check if we need to de-accelerate
            if( ((BaseType::vel * BaseType::vel) / BaseType::max_acc) >= fabs(setPoint - BaseType::pos) * PositionType(2.0) ) 
            {
                if(BaseType::vel < VelocityType(0.0)) 
                {
                    BaseType::pos += (BaseType::vel + BaseType::max_acc * dt) * dt;
                }
                else if(BaseType::vel > VelocityType(0.0)) 
                {
                    BaseType::pos += (BaseType::vel - BaseType::max_acc * dt) * dt;
                }
            }
            else {
                // We're not too close yet, so no need to de-accelerate. Check if we need to accelerate or maintain velocity.
                if(fabs(BaseType::vel) < BaseType::max_vel || (setPoint < BaseType::pos && BaseType::vel > VelocityType(0.0)) || (setPoint > BaseType::pos && BaseType::vel < VelocityType(0.0))) 
                {
                    // We need to accelerate, do so but check the maximum acceleration.
                    // Keep velocity constant at the maximum
                    VelocityType suggestedVelocity = VelocityType(0.0);
                    
                    if(setPoint > BaseType::pos) 
                    {
                        suggestedVelocity = BaseType::vel + BaseType::max_acc * dt;
                        
                        if (suggestedVelocity > BaseType::max_vel) 
                        {
                            suggestedVelocity = BaseType::max_vel;
                        }
                    }
                    else 
                    {
                        suggestedVelocity = BaseType::vel - BaseType::max_acc * dt;
                        
                        if(fabs(suggestedVelocity) > BaseType::max_vel) 
                        {
                            suggestedVelocity = -BaseType::max_vel;
                        }               
                    }
                    
                    BaseType::pos += suggestedVelocity * dt;
                }
                else 
                {
                    // Keep velocity constant at the maximum
                    if(setPoint > BaseType::pos) 
                    {
                        BaseType::pos += BaseType::max_vel * dt;
                    }
                    else 
                    {
                        BaseType::pos += -BaseType::max_vel * dt;
                    }
                }
            }
            
            
            BaseType::calcState(dt);
        }
        
        return BaseType::pos;
    }
};

/**
 * Constant Velocity Profile.
 */
template<typename PT, typename TT = PT, typename VT = PT, typename AT = PT>
class ConstantVelocityProfile : public VelocityProfileBase<PT,TT,VT,AT>
{
    typedef VelocityProfileBase<PT,TT,VT,AT> BaseType;
public:
    typedef PT PositionType;
    typedef TT TimeType;
    typedef VT VelocityType;
    typedef AT AccelerationType;
    
    inline ConstantVelocityProfile() : BaseType()
    {
        
    }
    
    inline ConstantVelocityProfile(VelocityType avel_max, AccelerationType aacc_max) : BaseType(avel_max, aacc_max)
    {
        
    }
    
    inline PositionType operator()(PositionType setPoint, TimeType dt, PositionType eps = std::numeric_limits<PositionType>::epsilon())
    {
        BaseType::is_finished = ( fabs(setPoint - BaseType::pos) < eps );
        
        if(!BaseType::is_finished)
        {
            BaseType::prev_pos = BaseType::pos;
            BaseType::prev_vel = BaseType::vel;
            
            VelocityType suggestedVelocity = (setPoint - BaseType::pos) / dt;
            
            if(suggestedVelocity > BaseType::max_vel) 
            {
                BaseType::pos += BaseType::max_vel * dt;
            }
            else if(suggestedVelocity < -BaseType::max_vel) 
            {
                BaseType::pos += -BaseType::max_vel * dt;
            }
            else 
            {
                BaseType::pos += suggestedVelocity * dt;
            }
            
            BaseType::calcState(dt);
        }
        
        return BaseType::pos;
    }
};
    
}

}

#endif // VISIONCORE_CONTROL_VELOCITY_PROFILE_HPP
