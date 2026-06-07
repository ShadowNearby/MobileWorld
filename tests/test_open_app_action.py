"""Tests for the `open_app` action in the general_e2e test flow.

`open_app` lets an agent launch an app directly by name instead of returning to
the home screen and swiping through launcher pages to find its icon. These tests
cover the full general_e2e path:

    model output  ->  parse_action / parse_response_to_action  ->  JSONAction
                  ->  AndroidController.launch_app  ->  adb monkey LAUNCHER

so the action is exercised as a first-class citizen of the test flow, not just
advertised in the prompt.
"""
from __future__ import annotations

import pytest

from mobile_world.agents.implementations.general_e2e_agent import (
    normalize_action_type,
    parse_action,
    parse_response_to_action,
)
from mobile_world.agents.utils.prompts import GENERAL_E2E_PROMPT_TEMPLATE
from mobile_world.runtime import controller as controller_mod
from mobile_world.runtime.controller import APP_LOWER_DICT, AndroidController
from mobile_world.runtime.utils.helpers import AdbResponse
from mobile_world.runtime.utils.models import (
    OPEN_APP,
    JSONAction,
    _ACTION_TYPES,
    available_app_names,
)


# --------------------------------------------------------------------------- #
# 1. The general_e2e prompt advertises open_app to the model.
# --------------------------------------------------------------------------- #
def test_general_e2e_prompt_advertises_open_app():
    rendered = GENERAL_E2E_PROMPT_TEMPLATE.render(scale_factor=1000, tools=None)
    # Action table row is present with the documented JSON shape.
    assert "open_app" in rendered
    assert '"action_type":"open_app"' in rendered.replace(" ", "")
    # A mandatory rule forces the model to use open_app instead of tapping icons.
    assert "Opening Apps (MANDATORY)" in rendered
    assert "MUST use the `open_app`" in rendered
    assert "FORBIDDEN" in rendered


def test_available_app_names_all_resolve_and_dedup():
    names = available_app_names()
    assert names, "available_app_names() should not be empty"
    # No case-insensitive duplicates.
    folded = [n.casefold() for n in names]
    assert len(folded) == len(set(folded))
    # Every advertised name must actually resolve via the launcher mapping
    # (launch_app matches on app_name.lower()).
    unresolved = [n for n in names if n.lower() not in APP_LOWER_DICT]
    assert not unresolved, f"names not in APP_LOWER_DICT: {unresolved}"


def test_prompt_injects_available_apps_when_provided():
    names = available_app_names()
    rendered = GENERAL_E2E_PROMPT_TEMPLATE.render(
        scale_factor=1000, tools=None, available_apps=", ".join(names)
    )
    assert "choose `app_name` from this list" in rendered
    assert "Maps" in rendered and "Chrome" in rendered
    # Package-id fallback is still documented alongside the list.
    assert "package id" in rendered


def test_prompt_omits_app_list_when_not_provided():
    rendered = GENERAL_E2E_PROMPT_TEMPLATE.render(scale_factor=1000, tools=None)
    # open_app row stays, but the injected name list does not appear.
    assert "open_app" in rendered
    assert "choose `app_name` from this list" not in rendered


# --------------------------------------------------------------------------- #
# 2. open_app is a valid, normalized action type.
# --------------------------------------------------------------------------- #
def test_open_app_is_valid_action_type():
    assert OPEN_APP == "open_app"
    assert OPEN_APP in _ACTION_TYPES
    assert normalize_action_type("open_app") == "open_app"
    # JSONAction validation accepts it with an app_name payload.
    action = JSONAction(action_type="open_app", app_name="Maps")
    assert action.action_type == "open_app"
    assert action.app_name == "Maps"


# --------------------------------------------------------------------------- #
# 3. The general_e2e parser turns model output into an open_app JSONAction.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("app_name", ["Maps", "com.google.android.apps.maps"])
def test_general_e2e_parses_open_app(app_name):
    model_output = (
        "Thought: I should open the app directly instead of searching for its icon.\n"
        f'Action: {{"action_type": "open_app", "app_name": "{app_name}"}}'
    )
    thought, action_str = parse_action(model_output)
    assert "directly" in thought

    parsed = parse_response_to_action(action_str, image_width=1080, image_height=2424)
    action = JSONAction(**parsed)
    assert action.action_type == "open_app"
    assert action.app_name == app_name


# --------------------------------------------------------------------------- #
# 4. AndroidController.launch_app dispatch (no real device).
# --------------------------------------------------------------------------- #
class _FakeAdb:
    """Records adb invocations and answers them based on the command string."""

    def __init__(self, installed: set[str]):
        self.installed = installed
        self.commands: list[str] = []

    def __call__(self, command: str) -> AdbResponse:
        self.commands.append(command)
        if "pm list packages" in command:
            pkg = command.split()[-1]
            if pkg in self.installed:
                return AdbResponse(success=True, output=f"package:{pkg}\n")
            return AdbResponse(success=True, output="")
        if "monkey -p" in command:
            return AdbResponse(success=True, output="Events injected: 1")
        return AdbResponse(success=True, output="")

    def last_monkey_package(self) -> str | None:
        for cmd in reversed(self.commands):
            if "monkey -p" in cmd:
                return cmd.split("monkey -p", 1)[1].split()[0]
        return None


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(AndroidController, "get_device_size", lambda self: (1080, 2424))
    return AndroidController(device="test-device")


def test_launch_app_known_name_uses_mapping(controller, monkeypatch):
    fake = _FakeAdb(installed=set())
    monkeypatch.setattr(controller_mod, "execute_adb", fake)

    ret = controller.launch_app("Maps")

    assert ret.success
    assert fake.last_monkey_package() == APP_LOWER_DICT["maps"]


def test_launch_app_raw_package_fallback(controller, monkeypatch):
    # A package id that is NOT in APP_LOWER_DICT but IS installed on device.
    pkg = "com.example.thirdparty"
    assert pkg not in APP_LOWER_DICT
    fake = _FakeAdb(installed={pkg})
    monkeypatch.setattr(controller_mod, "execute_adb", fake)

    ret = controller.launch_app(pkg)

    assert ret.success
    assert fake.last_monkey_package() == pkg


def test_launch_app_unknown_name_fails(controller, monkeypatch):
    fake = _FakeAdb(installed=set())
    monkeypatch.setattr(controller_mod, "execute_adb", fake)

    ret = controller.launch_app("com.not.installed.app")

    assert not ret.success
    # Never issued a launch for an app we couldn't resolve.
    assert fake.last_monkey_package() is None


def test_launch_app_none_fails(controller, monkeypatch):
    fake = _FakeAdb(installed=set())
    monkeypatch.setattr(controller_mod, "execute_adb", fake)

    ret = controller.launch_app(None)

    assert not ret.success


# --------------------------------------------------------------------------- #
# 5. Device-aware app list: only installed + known apps, English preferred.
# --------------------------------------------------------------------------- #
def test_installed_app_names_intersects_and_prefers_english(controller, monkeypatch):
    installed = [
        "com.google.android.apps.maps",  # APP_DICT "Maps" / COMMON "地图" -> Maps
        "com.android.chrome",            # Chrome
        "com.google.android.apps.bard",  # Gemini
        "com.some.unknown.app",          # not in any mapping -> excluded
    ]

    def fake(cmd: str) -> AdbResponse:
        if cmd.strip().endswith("pm list packages"):
            return AdbResponse(
                success=True, output="".join(f"package:{p}\n" for p in installed)
            )
        return AdbResponse(success=True, output="")

    monkeypatch.setattr(controller_mod, "execute_adb", fake)
    names = controller.installed_app_names()

    assert "Maps" in names and "Chrome" in names and "Gemini" in names
    assert "地图" not in names  # English name preferred for the maps package
    assert "com.some.unknown.app" not in names  # unknown package omitted
    assert all(n.lower() in APP_LOWER_DICT for n in names)


def test_installed_app_names_empty_on_failure(controller, monkeypatch):
    monkeypatch.setattr(
        controller_mod,
        "execute_adb",
        lambda cmd: AdbResponse(success=False, error="boom"),
    )
    assert controller.installed_app_names() == []


def test_agent_set_available_apps_semantics():
    from mobile_world.agents.implementations.general_e2e_agent import GeneralE2EAgentMCP

    agent = GeneralE2EAgentMCP(
        model_name="qwen", llm_base_url="http://localhost:1/v1", api_key="x"
    )
    assert agent.available_apps is None  # falls back to static list at render
    agent.set_available_apps(["Maps", "Gemini"])
    assert agent.available_apps == ["Maps", "Gemini"]
    agent.set_available_apps([])  # empty -> None (fall back to static)
    assert agent.available_apps is None
    agent.set_available_apps(None)
    assert agent.available_apps is None
