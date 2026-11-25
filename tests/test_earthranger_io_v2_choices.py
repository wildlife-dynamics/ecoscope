from unittest.mock import patch

import pytest

from ecoscope.io.earthranger import EarthRangerIO

V2 = "https://x.pamdas.org/api/v2.0"


def _io():
    er = EarthRangerIO.__new__(EarthRangerIO)
    er.server = "https://x.pamdas.org"
    er.service_root = "https://x.pamdas.org/api/v1.0"
    return er


@pytest.mark.parametrize(
    "ref",
    [
        "/api/v2.0/schemas/choices.json?field=herd",  # server path, as mep-dev sends it
        "https://x.pamdas.org/api/v2.0/schemas/choices.json?field=herd",  # full URL
        "schemas/choices.json?field=herd",  # relative
    ],
)
def test_choice_refs_of_every_form_resolve(ref):
    responses = {
        f"{V2}/activity/eventtypes/t/schema": {"json": {"properties": {"herd": {"anyOf": [{"$ref": ref}]}}}},
        f"{V2}/schemas/choices.json?field=herd": {"oneOf": [{"const": "bull", "title": "Bull"}]},
    }

    def fake_get(self, path, **kwargs):  # resolves URLs the way erclient's _get does
        return responses[path if path.startswith("http") else self._er_url(path)]

    with patch.object(EarthRangerIO, "_get", autospec=True, side_effect=fake_get):
        assert _io().get_choices_from_v2_event_type("t", "herd") == {"bull": "Bull"}
