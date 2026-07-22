// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <unitree/common/thread/recurrent_thread.hpp>
#include <chrono>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }
    }

    void start(const std::string& initial_state = "Passive",
               const std::string& automatic_next_state = "",
               double automatic_transition_delay_s = 3.0,
               const std::string& automatic_second_state = "",
               double automatic_second_delay_s = 10.0)
    {
        currentState = nullptr;
        for (auto& state : states)
        {
            if (state->getStateString() == initial_state)
            {
                currentState = state;
                break;
            }
        }
        if (!currentState)
        {
            throw std::runtime_error("FSM: Unknown initial state " + initial_state);
        }
        currentState->enter();
        automatic_transitions_.clear();
        if (!automatic_next_state.empty()) {
            automatic_transitions_.push_back(
                {automatic_transition_delay_s, automatic_next_state});
        }
        if (!automatic_second_state.empty()) {
            automatic_transitions_.push_back(
                {automatic_second_delay_s, automatic_second_state});
        }
        std::sort(automatic_transitions_.begin(), automatic_transitions_.end());
        automatic_transition_index_ = 0;
        automatic_transition_start_ = std::chrono::steady_clock::now();

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }
    
    ~CtrlFSM()
    {
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();
        
        // Check if need to change state
        int nextStateMode = 0;
        for(int i(0); i<currentState->registered_checks.size(); i++)
        {
            if(currentState->registered_checks[i].first())
            {
                nextStateMode = currentState->registered_checks[i].second;
                break;
            }
        }

        if (nextStateMode == 0 &&
            automatic_transition_index_ < automatic_transitions_.size())
        {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - automatic_transition_start_).count();
            const auto& transition =
                automatic_transitions_[automatic_transition_index_];
            if (elapsed >= transition.first)
            {
                const std::string requested_state = transition.second;
                ++automatic_transition_index_;
                bool found = false;
                for (auto& state : states)
                {
                    if (state->getStateString() == requested_state)
                    {
                        nextStateMode = state->getState();
                        spdlog::info("FSM: Automatic simulation-test transition requested: {}",
                                     requested_state);
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    spdlog::error("FSM: Unknown automatic transition state {}",
                                  requested_state);
                }
            }
        }

        if(nextStateMode != 0 && !currentState->isState(nextStateMode))
        {
            for(auto & state : states)
            {
                if(state->isState(nextStateMode))
                {
                    spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
                    currentState->exit();
                    currentState = state;
                    currentState->enter();
                    break;
                }
            }
        }
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
    std::vector<std::pair<double, std::string>> automatic_transitions_;
    std::size_t automatic_transition_index_ = 0;
    std::chrono::steady_clock::time_point automatic_transition_start_;
};
