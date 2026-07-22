#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include "joystick/joystick.h"
#include <memory>

// Simulation-only request consumed by main.cc.  The physical controller never
// reads this flag, so the test hook cannot command a real winch.
inline std::atomic<bool> virtual_hoist_requested{false};


class XBoxJoystick : public unitree::common::UnitreeJoystick
{
public:
    XBoxJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[6]);
        start(js_->button_[7]);
        LB(js_->button_[4]);
        RB(js_->button_[5]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[2]);
        Y(js_->button_[3]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[2] > 0);
        RT(js_->axis_[5] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[3]) / max_value_);
        ry(-double(js_->axis_[4]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


class SwitchJoystick : public unitree::common::UnitreeJoystick
{
public:
    SwitchJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[10]);
        start(js_->button_[11]);
        LB(js_->button_[6]);
        RB(js_->button_[7]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[3]);
        Y(js_->button_[4]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[5] > 0);
        RT(js_->axis_[4] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[2]) / max_value_);
        ry(-double(js_->axis_[3]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


// Standard Linux joystick mapping used by a USB-connected DualShock 4.
// Unitree's logical A/B/X/Y names are mapped to Cross/Circle/Square/Triangle
// so the same FSM expressions can be shared with the physical G1 controller.
class PS4Joystick : public unitree::common::UnitreeJoystick
{
public:
    PS4Joystick(std::string device, int bits = 15)
    : unitree::common::UnitreeJoystick()
    {
        js_ = std::make_unique<Joystick>(device);
        if(!js_->isFound()) {
            std::cout << "Error: Joystick open failed." << std::endl;
            exit(1);
        }
        max_value_ = 1 << (bits - 1);
    }

    void update() override
    {
        js_->getState();
        back(js_->button_[8]);       // Share
        start(js_->button_[9]);      // Options
        LB(js_->button_[4]);         // L1
        RB(js_->button_[5]);         // R1
        A(js_->button_[1]);          // Cross
        B(js_->button_[2]);          // Circle
        X(js_->button_[0]);          // Square
        Y(js_->button_[3]);          // Triangle
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);

        // DS4 exposes L2/R2 as buttons on the usual joydev mapping.  Also
        // accept their analogue axes for adapters which omit trigger buttons.
        LT(js_->button_[6] || js_->axis_[3] > 0);
        RT(js_->button_[7] || js_->axis_[4] > 0);

        const bool hoist_combo =
            (js_->button_[7] || js_->axis_[4] > 0) && js_->button_[3];
        if (hoist_combo && !previous_hoist_combo_)
        {
            virtual_hoist_requested.store(true, std::memory_order_release);
            std::cout << "[HOIST] PS4 R2+Triangle request received\n";
        }
        previous_hoist_combo_ = hoist_combo;

        lx(filtered_axis(0));
        ly(-filtered_axis(1));
        rx(filtered_axis(2));
        ry(-filtered_axis(5));
    }
private:
    double filtered_axis(int id) const
    {
        constexpr double deadzone = 0.10;
        const double value = std::clamp(
            double(js_->axis_[id]) / max_value_, -1.0, 1.0
        );
        const double magnitude = std::abs(value);
        if (magnitude <= deadzone)
        {
            return 0.0;
        }
        return std::copysign(
            (magnitude - deadzone) / (1.0 - deadzone), value
        );
    }

    std::unique_ptr<Joystick> js_;
    int max_value_;
    bool previous_hoist_combo_ = false;
};
