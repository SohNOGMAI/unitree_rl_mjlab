#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_Crouch.h"
#include "State_FixedPosture.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub =
        std::make_shared<unitree::robot::g1::subscription::LowCmd>();

    usleep(0.2 * 1e6);

    if (!lowcmd_sub->isTimeout())
    {
        spdlog::critical(
            "The other process is using the lowcmd channel, "
            "please close it first."
        );

        unitree::robot::go2::shutdown();

        // 二重にLowCmdを送るのは危険なので、必ず終了する。
        std::exit(EXIT_FAILURE);
    }

    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();

    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // コマンドライン引数とconfig.yamlを読み込む。
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDSの設定。
    const auto network = vm["network"].as<std::string>();
    const auto domain_id = vm["domain-id"].as<int>();

    const auto initial_state =
        vm["initial-state"].as<std::string>();

    const auto auto_transition_state =
        vm["auto-transition-state"].as<std::string>();

    const auto auto_transition_delay_s =
        vm["auto-transition-delay-s"].as<double>();

    const auto auto_second_transition_state =
        vm["auto-second-transition-state"].as<std::string>();

    const auto auto_second_transition_delay_s =
        vm["auto-second-transition-delay-s"].as<double>();

    // 自動状態遷移やPassive以外からの起動は、
    // MuJoCoのloopback通信でのみ許可する。
    // 実機での自動遷移を防止するための安全確認。
    if ((initial_state != "Passive"
         || !auto_transition_state.empty()
         || !auto_second_transition_state.empty())
        && network != "lo")
    {
        spdlog::critical(
            "Automatic/non-Passive startup is simulation-only "
            "and requires --network lo."
        );

        return EXIT_FAILURE;
    }

    unitree::robot::ChannelFactory::Instance()->Init(
        domain_id,
        network
    );

    init_fsm_state();

    // G1 29DoFモードを指定する。
    FSMState::lowcmd->msg_.mode_machine() = 5;

    if (!FSMState::lowcmd->check_mode_machine(FSMState::lowstate))
    {
        spdlog::critical("Unmatched robot type.");
        return EXIT_FAILURE;
    }

    // FSMを初期化して、必ずPassiveから開始する。
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);

    fsm->start(
        initial_state,
        auto_transition_state,
        auto_transition_delay_s,
        auto_second_transition_state,
        auto_second_transition_delay_s
    );

    std::cout
        << "Input: Unitree wireless controller carried in G1 LowState\n";

    std::cout
        << "Unitree: [LT + D-pad Up] "
        << "Passive -> zero-command stand policy\n";

    std::cout
        << "Unitree: [RT + B]        "
        << "stand/walking policy -> crouch policy\n";

    std::cout
        << "Unitree: [LT + X]        "
        << "Passive -> FixStand (pose test only)\n";

    std::cout
        << "Unitree: [RT + A]        "
        << "stand/FixStand -> joystick walking policy\n";

    std::cout
        << "Unitree: [RT + X]        "
        << "crouch/fixed posture -> FixStand\n";

    std::cout
        << "Unitree: [RT + Y]        "
        << "crouch -> fixed suspension posture "
        << "(HARNESS ONLY)\n";

    std::cout
        << "Unitree: [RB + Y]        "
        << "supported stand -> fixed suspension posture "
        << "(HARNESS ONLY)\n";

    std::cout
        << "Unitree: [LT + B]        "
        << "software stop -> Passive\n";

    std::cout
        << "PC terminal: [SPACE]     "
        << "software stop -> Passive\n";

    std::cout
        << "PC terminal: [1]         "
        << "Passive -> zero-command stand policy\n";

    std::cout
        << "PC terminal: [2]         "
        << "stand/walking policy -> crouch policy\n";

    std::cout
        << "PC terminal: [Shift+F]   "
        << "supported stand/crouch -> fixed suspension posture "
        << "(HARNESS ONLY)\n";

    std::cout
        << "PC terminal: [4]         "
        << "Passive -> FixStand or crouch/fixed posture -> FixStand\n";

    std::cout
        << "PC terminal: [5]         "
        << "stand/FixStand -> joystick walking policy\n";

    std::cout
        << "WARNING: software Passive is not an "
        << "independent hardware E-stop.\n";

    while (true)
    {
        sleep(1);
    }

    return 0;
}
