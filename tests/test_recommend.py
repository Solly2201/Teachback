from app.recommend.rules import GENERIC_ACTIVITIES, recommend
from app.states import STATE_KEYS


def test_every_state_has_a_generic_activity():
    for key in STATE_KEYS:
        assert key in GENERIC_ACTIVITIES


def test_generic_fallback():
    rec = recommend(2)
    assert rec["state_key"] == "struggling"
    assert rec["activity"]["kind"] == "guided_practice"


def test_topic_activity_preferred():
    activities = [{"title": "Custom drill", "description": "d", "kind": "guided_practice",
                   "target_state": "struggling"}]
    rec = recommend(2, activities)
    assert rec["activity"]["title"] == "Custom drill"


def test_misconception_note_attached():
    rec = recommend(3, None, ["Gradients directly change the weights"])
    assert any("Gradients directly change the weights" in n for n in rec["notes"])
