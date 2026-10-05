from __future__ import annotations

import subprocess
from pathlib import Path

from game_balatro.balatrobot import BalatroBotClient, snapshot_from_gamestate
from game_balatro.models import BalatroDecision, ScreenType
from game_balatro.solver import BalatroSolver


def test_balatrobot_state_maps_to_solver_snapshot() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SELECTING_HAND",
            "round_num": 2,
            "ante_num": 1,
            "money": 9,
            "round": {
                "hands_left": 3,
                "discards_left": 2,
                "chips": 125,
                "reroll_cost": 5,
            },
            "blinds": {
                "small": {
                    "status": "CURRENT",
                    "name": "Small Blind",
                    "score": 300,
                }
            },
            "hand": {
                "cards": [
                    {
                        "label": "Ten of Hearts",
                        "set": "DEFAULT",
                        "value": {"rank": "T", "suit": "H"},
                        "modifier": {},
                        "state": {"highlight": False, "debuff": False},
                        "cost": {"buy": 0},
                    },
                    {
                        "label": "Ace of Spades",
                        "set": "DEFAULT",
                        "value": {"rank": "A", "suit": "S"},
                        "modifier": {"enhancement": "BONUS"},
                        "state": {"highlight": True, "debuff": False},
                        "cost": {"buy": 0},
                    },
                ]
            },
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )

    assert snapshot.screen_type == ScreenType.PLAY
    assert snapshot.ante == 1
    assert snapshot.round_no == 2
    assert snapshot.current_score == 125
    assert snapshot.target_score == 300
    assert snapshot.hands_left == 3
    assert snapshot.discards_left == 2
    assert snapshot.hand[0].rank == "10"
    assert snapshot.hand[0].suit == "H"
    assert snapshot.hand[1].selected is True
    assert snapshot.hand[1].enhancement == "BONUS"
    assert snapshot.play_button is not None
    assert snapshot.play_button.visible is False
    assert snapshot.metadata["source"] == "balatrobot"


def test_balatrobot_shop_keeps_real_api_indices() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "money": 20,
            "round": {"reroll_cost": 5},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {
                "cards": [
                    {
                        "label": "Joker A",
                        "set": "JOKER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "vouchers": {
                "cards": [
                    {
                        "label": "Voucher A",
                        "set": "VOUCHER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 10},
                    }
                ]
            },
            "packs": {
                "cards": [
                    {
                        "label": "Arcana Pack",
                        "set": "BOOSTER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "pack": {"cards": []},
        }
    )

    assert [(item.api_group, item.api_index) for item in snapshot.shop_items] == [
        ("card", 0),
        ("voucher", 0),
        ("pack", 0),
    ]
    assert [item.kind for item in snapshot.shop_items] == [
        "joker",
        "voucher",
        "pack",
    ]


def test_balatrobot_maps_decisions_without_mouse_coordinates() -> None:
    client = BalatroBotClient(
        base_url="http://127.0.0.1:12346",
        deck="RED",
        stake="WHITE",
    )
    play_snapshot = snapshot_from_gamestate(
        {
            "state": "SELECTING_HAND",
            "round": {"hands_left": 4, "discards_left": 3, "chips": 0},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    assert client.command_for_decision(
        play_snapshot,
        BalatroDecision(
            action="play_hand",
            reason="test",
            card_indices=(0, 2, 4),
        ),
    ) == ("play", {"cards": [0, 2, 4]})

    shop_snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "round": {},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    assert client.command_for_decision(
        shop_snapshot,
        BalatroDecision(
            action="click",
            reason="buy",
            metadata={
                "candidate_type": "shop_buy",
                "api_group": "voucher",
                "api_index": 1,
            },
        ),
    ) == ("buy", {"voucher": 1})


def test_round_eval_maps_to_cash_out_screen() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "ROUND_EVAL",
            "round": {"chips": 500},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    assert snapshot.screen_type == ScreenType.ROUND_EVAL


def test_api_shop_candidates_keep_leave_and_reroll_separate() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "money": 20,
            "round": {"reroll_cost": 5},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {
                "cards": [
                    {
                        "label": "Joker A",
                        "set": "JOKER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "vouchers": {"cards": []},
            "packs": {"cards": []},
            "pack": {"cards": []},
        }
    )
    candidates = BalatroSolver(reserve_money=5).rank_candidates(
        snapshot,
        top_n=6,
    )
    kinds = {
        str(decision.metadata.get("candidate_type") or "")
        for decision in candidates
    }

    assert "shop_buy" in kinds
    assert "shop_leave" in kinds
    assert "shop_reroll" in kinds


def test_api_shop_booster_baseline_keeps_buy_group_and_index() -> None:
    snapshot = snapshot_from_gamestate(
        {
            "state": "SHOP",
            "money": 20,
            "round": {"reroll_cost": 5},
            "blinds": {},
            "hand": {"cards": []},
            "jokers": {"cards": []},
            "shop": {"cards": []},
            "vouchers": {"cards": []},
            "packs": {
                "cards": [
                    {
                        "label": "Arcana Pack",
                        "set": "BOOSTER",
                        "value": {},
                        "modifier": {},
                        "state": {},
                        "cost": {"buy": 4},
                    }
                ]
            },
            "pack": {"cards": []},
        }
    )
    decision = BalatroSolver(reserve_money=5).decide(snapshot)

    assert decision.metadata["candidate_type"] == "shop_buy"
    assert decision.metadata["api_group"] == "pack"
    assert decision.metadata["api_index"] == 0



def test_balatrobot_ensure_server_skips_spawn_when_api_is_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = BalatroBotClient()
    monkeypatch.setattr(
        client,
        "health",
        lambda **_kwargs: True,
    )

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("API 已正常时不应该启动 uvx")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    assert client.ensure_server(logs_dir=tmp_path) == "already_running"


def test_balatrobot_ensure_server_directly_starts_balatro_on_windows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = BalatroBotClient()
    checks = iter((False, True))
    monkeypatch.setattr(
        client,
        "health",
        lambda **_kwargs: next(checks),
    )
    monkeypatch.setattr(
        client,
        "_patch_preserve_user_settings",
        lambda: "patched",
    )
    monkeypatch.setattr(
        client,
        "_direct_launch_command",
        lambda: (
            ["C:/Games/Balatro/Balatro.exe"],
            {
                "BALATROBOT_HOST": "127.0.0.1",
                "BALATROBOT_PORT": "12346",
                "BALATROBOT_AUDIO": "1",
                "NO_PROXY": "127.0.0.1,localhost,::1",
                "no_proxy": "127.0.0.1,localhost,::1",
            },
        ),
    )
    captured: dict[str, object] = {}

    class DummyProcess:
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert (
        client.ensure_server(
            logs_dir=tmp_path,
            startup_timeout_seconds=3.0,
        )
        == "started"
    )
    assert captured["command"] == [
        "C:/Games/Balatro/Balatro.exe"
    ]
    env = captured["kwargs"]["env"]
    assert env["BALATROBOT_HOST"] == "127.0.0.1"
    assert env["BALATROBOT_PORT"] == "12346"
    assert env["BALATROBOT_AUDIO"] == "1"
    assert "127.0.0.1" in env["NO_PROXY"]
    assert (tmp_path / "balatro_direct.log").exists()



def test_windows_launch_path_prefers_winmm_over_version(
    tmp_path: Path,
) -> None:
    game_dir = tmp_path / "Balatro"
    game_dir.mkdir()
    game = game_dir / "Balatro.exe"
    winmm = game_dir / "winmm.dll"
    version = game_dir / "version.dll"
    game.write_bytes(b"exe")
    winmm.write_bytes(b"lovely-new")
    version.write_bytes(b"lovely-old")

    client = BalatroBotClient(
        game_executable_path=str(game),
    )
    resolved_game, resolved_lovely = (
        client._resolve_windows_launch_paths()
    )

    assert resolved_game == game.resolve()
    assert resolved_lovely == winmm.resolve()


def test_windows_launch_path_falls_back_to_version_dll(
    tmp_path: Path,
) -> None:
    game_dir = tmp_path / "Balatro"
    game_dir.mkdir()
    game = game_dir / "Balatro.exe"
    version = game_dir / "version.dll"
    game.write_bytes(b"exe")
    version.write_bytes(b"lovely-old")

    client = BalatroBotClient(
        game_executable_path=str(game),
    )
    resolved_game, resolved_lovely = (
        client._resolve_windows_launch_paths()
    )

    assert resolved_game == game.resolve()
    assert resolved_lovely == version.resolve()


def test_windows_launch_args_pass_balatro_and_lovely_paths(
    tmp_path: Path,
) -> None:
    game_dir = tmp_path / "Balatro"
    game_dir.mkdir()
    game = game_dir / "Balatro.exe"
    lovely = game_dir / "winmm.dll"
    game.write_bytes(b"exe")
    lovely.write_bytes(b"lovely")

    client = BalatroBotClient(
        game_executable_path=str(game),
        lovely_path=str(lovely),
    )

    assert client._windows_launch_args() == [
        "--love-path",
        str(game.resolve()),
        "--lovely-path",
        str(lovely.resolve()),
        "--platform",
        "windows",
    ]


def test_direct_launch_passes_balatrobot_environment(
    tmp_path: Path,
) -> None:
    game_dir = tmp_path / "Balatro"
    game_dir.mkdir()
    game = game_dir / "Balatro.exe"
    lovely = game_dir / "winmm.dll"
    game.write_bytes(b"exe")
    lovely.write_bytes(b"lovely")

    client = BalatroBotClient(
        base_url="http://127.0.0.1:23456",
        game_executable_path=str(game),
        lovely_path=str(lovely),
        audio=True,
    )
    command, env = client._direct_launch_command()

    assert command == [str(game.resolve())]
    assert env["BALATROBOT_HOST"] == "127.0.0.1"
    assert env["BALATROBOT_PORT"] == "23456"
    assert env["BALATROBOT_AUDIO"] == "1"
    assert "localhost" in env["NO_PROXY"]
    assert "::1" in env["no_proxy"]



def test_balatrobot_audio_can_be_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    game_dir = tmp_path / "Balatro"
    game_dir.mkdir()
    game = game_dir / "Balatro.exe"
    lovely = game_dir / "winmm.dll"
    game.write_bytes(b"exe")
    lovely.write_bytes(b"lovely")

    client = BalatroBotClient(
        audio=False,
        game_executable_path=str(game),
        lovely_path=str(lovely),
    )
    command, env = client._direct_launch_command()

    assert command == [str(game.resolve())]
    assert env["BALATROBOT_AUDIO"] == "0"



def _write_balatrobot_settings_fixture(mod_dir: Path) -> Path:
    settings = mod_dir / "src" / "lua" / "settings.lua"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        """
local function configure_settings()
  G.SETTINGS.SOUND.volume = 0
  G.SETTINGS.SOUND.music_volume = 0
  G.SETTINGS.SOUND.game_sounds_volume = 0
  G.F_SOUND_THREAD = false
  G.F_MUTE = true
  G.SETTINGS.GAMESPEED = BB_SETTINGS.gamespeed
  G.SETTINGS.GRAPHICS.shadows = "Off"
  G.SETTINGS.GRAPHICS.bloom = 0
  G.SETTINGS.GRAPHICS.crt = 0
  G.SETTINGS.reduced_motion = true
  G.SETTINGS.screenshake = false
  G.SETTINGS.WINDOW.vsync = 0
end

--- Configures headless mode
local function configure_headless()
end

local function configure_audio()
  G.SETTINGS.SOUND = G.SETTINGS.SOUND or {}
  G.SETTINGS.SOUND.volume = 50
  G.SETTINGS.SOUND.music_volume = 100
  G.SETTINGS.SOUND.game_sounds_volume = 100
  G.F_MUTE = false
  G.F_SOUND_THREAD = true
end

--- Initializes and applies
BB_SETTINGS.setup = function()
end
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return settings


def test_preserve_user_settings_patch_is_idempotent(
    tmp_path: Path,
) -> None:
    mod_dir = tmp_path / "balatrobot"
    settings = _write_balatrobot_settings_fixture(mod_dir)
    client = BalatroBotClient(
        mod_path=str(mod_dir),
        preserve_user_settings=True,
        audio=True,
    )

    assert client._patch_preserve_user_settings() == "patched"
    patched = settings.read_text(encoding="utf-8")
    assert "AIAGENT_PRESERVE_USER_SETTINGS_V2" in patched
    assert "G.SETTINGS.GAMESPEED = BB_SETTINGS.gamespeed" not in patched
    assert 'G.SETTINGS.GRAPHICS.shadows = "Off"' not in patched
    assert "G.SETTINGS.WINDOW.vsync = 0" not in patched
    assert "G.F_SOUND_THREAD = true" in patched
    assert settings.with_name("settings.lua.aiagent.bak").exists()

    assert client._patch_preserve_user_settings() == "already_patched"


def test_preserve_user_settings_patch_can_be_disabled(
    tmp_path: Path,
) -> None:
    mod_dir = tmp_path / "balatrobot"
    settings = _write_balatrobot_settings_fixture(mod_dir)
    original = settings.read_text(encoding="utf-8")
    client = BalatroBotClient(
        mod_path=str(mod_dir),
        preserve_user_settings=False,
    )

    assert client._patch_preserve_user_settings() == "disabled"
    assert settings.read_text(encoding="utf-8") == original
    assert not settings.with_name("settings.lua.aiagent.bak").exists()



def test_balatrobot_local_rpc_forces_proxy_bypass(
    monkeypatch,
) -> None:
    client = BalatroBotClient(
        base_url="http://127.0.0.1:12346",
    )
    called: dict[str, object] = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return (
                b'{"jsonrpc":"2.0","result":{"status":"ok"},"id":1}'
            )

    class DummyOpener:
        @staticmethod
        def open(request, timeout):
            called["url"] = request.full_url
            called["timeout"] = timeout
            return DummyResponse()

    def fake_build_opener(handler):
        called["handler"] = handler
        return DummyOpener()

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("localhost RPC 不应该走默认 urlopen/系统代理")

    monkeypatch.setattr(
        "game_balatro.balatrobot.build_opener",
        fake_build_opener,
    )
    monkeypatch.setattr(
        "game_balatro.balatrobot.urlopen",
        fail_urlopen,
    )

    result = client.call("health")

    assert result["status"] == "ok"
    assert called["url"] == "http://127.0.0.1:12346"
