#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;
    std::string initial_keyframe;
    bool no_joystick = false;
    bool auto_start_on_control = false;
    double auto_start_control_delay_s = 1.0;
    bool print_robot_diagnostics = false;

    // Simulation-only overhead cable used to validate the transition from the
    // crouch policy to the fixed suspension posture.  This is deliberately
    // opt-in and is never consumed by the real-robot controller.
    bool virtual_hoist = false;
    bool virtual_hoist_gamepad_trigger = false;
    bool virtual_hoist_pose_guide = false;
    double virtual_hoist_start_s = 11.0;
    double virtual_hoist_anchor_x = -0.14;
    double virtual_hoist_anchor_y = 0.0;
    double virtual_hoist_anchor_z = 2.4;
    double virtual_hoist_reel_speed_m_s = 0.05;
    double virtual_hoist_reel_distance_m = 0.20;
    double virtual_hoist_max_tension_n = 480.0;
    double virtual_hoist_tension_rate_n_s = 600.0;
    double virtual_hoist_stiffness_n_m = 2500.0;
    double virtual_hoist_damping_n_s_m = 250.0;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            initial_keyframe = cfg["initial_keyframe"]
                ? cfg["initial_keyframe"].as<std::string>() : "";
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
        ("no-joystick", po::bool_switch(&config.no_joystick),
         "disable the simulated wireless controller")
        ("auto-start-on-control", po::bool_switch(&config.auto_start_on_control),
         "start physics after a controller sends nonzero joint gains")
        ("auto-start-control-delay-s",
         po::value<double>(&config.auto_start_control_delay_s)->default_value(1.0),
         "wall-clock settling delay after detecting controller commands")
        ("diagnostics", po::bool_switch(&config.print_robot_diagnostics),
         "print base position and attitude during simulation")
        ("virtual-hoist", po::bool_switch(&config.virtual_hoist),
         "enable the simulation-only overhead cable")
        ("virtual-hoist-gamepad-trigger",
         po::bool_switch(&config.virtual_hoist_gamepad_trigger),
         "wait for PS4 R2+Triangle before starting the virtual cable")
        ("virtual-hoist-pose-guide",
         po::bool_switch(&config.virtual_hoist_pose_guide),
         "simulation fixture: hold horizontal COM position and yaw while hoisting")
        ("virtual-hoist-start-s",
         po::value<double>(&config.virtual_hoist_start_s)->default_value(11.0),
         "simulation time at which cable reeling starts")
        ("virtual-hoist-anchor-x",
         po::value<double>(&config.virtual_hoist_anchor_x)->default_value(-0.14),
         "overhead anchor world x position")
        ("virtual-hoist-anchor-y",
         po::value<double>(&config.virtual_hoist_anchor_y)->default_value(0.0),
         "overhead anchor world y position")
        ("virtual-hoist-anchor-z",
         po::value<double>(&config.virtual_hoist_anchor_z)->default_value(2.4),
         "overhead anchor world z position")
        ("virtual-hoist-reel-speed",
         po::value<double>(&config.virtual_hoist_reel_speed_m_s)->default_value(0.05),
         "cable command shortening speed in m/s")
        ("virtual-hoist-reel-distance",
         po::value<double>(&config.virtual_hoist_reel_distance_m)->default_value(0.20),
         "total cable command shortening in metres")
        ("virtual-hoist-max-tension",
         po::value<double>(&config.virtual_hoist_max_tension_n)->default_value(480.0),
         "simulation cable tension limit in newtons")
        ("virtual-hoist-tension-rate",
         po::value<double>(&config.virtual_hoist_tension_rate_n_s)->default_value(600.0),
         "simulation cable tension slew limit in N/s")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (config.no_joystick)
    {
        config.use_joystick = 0;
    }
    if (config.virtual_hoist && config.enable_elastic_band == 1)
    {
        std::cerr << "--virtual-hoist cannot be combined with enable_elastic_band=1\n";
        exit(EXIT_FAILURE);
    }
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}
