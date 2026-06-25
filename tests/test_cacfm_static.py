from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_flux_training_exposes_cacfm_rl_and_dmd_components():
    text = read("FLUX/train_tdd_adv.py")

    assert "class QLearning" in text
    assert "def compute_flux_distribution_matching_loss" in text
    assert 'parser.add_argument("--dmd_loss"' in text
    assert 'parser.add_argument("--dmd_weight", default=0.5' in text
    assert "def get_decayed_epsilon" in text
    assert "agent.update(state, action, reward, next_state)" in text
    assert "args.dmd_weight * compute_flux_distribution_matching_loss" in text
    dmd_call = text.split("args.dmd_weight * compute_flux_distribution_matching_loss", 1)[0].split("if args.dmd_loss:", 1)[1]
    assert "disable_adapters()" not in dmd_call
    assert "real_transformer.disable_adapters()" in text


def test_sdxl_reward_and_epsilon_schedule_match_paper_description():
    text = read("SDXL/train_pcm_base_model_sdxl_adv_RL.py")

    assert 'parser.add_argument("--RL_epsilon_start", default=1.0' in text
    assert 'parser.add_argument("--RL_epsilon_final", default=0.1' in text
    assert 'parser.add_argument("--RL_epsilon_decay_steps", default=20000' in text
    assert 'parser.add_argument("--dmd_weight", default=0.5' in text
    assert "def get_decayed_epsilon" in text
    assert "reward = args.reward_scale * (baseline_value - pcm_loss_value)" in text
    assert "agent.update(state, action, reward, next_state)" in text
    assert "agent.update(state, action, g_loss" not in text


def test_training_launchers_use_paper_dmd_weight():
    sdxl = read("SDXL/train_pcm_base_model_sdxl_RL_dmd.sh")
    flux = read("FLUX/train_tdd_adv.sh")

    assert "DMD_WEIGHT=0.5" in sdxl
    assert "DMD_WEIGHT=0.5" in flux
    assert "--dmd_weight=$DMD_WEIGHT" in flux
    assert "--dmd_loss" in flux
