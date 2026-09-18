from ecoscope.platform.tasks.config import set_float_var, set_optional_string_var


def test_set_optional_string_var_defaults_to_empty():
    assert set_optional_string_var() == ""


def test_set_optional_string_var_passthrough():
    assert set_optional_string_var(var="Number of Animals") == "Number of Animals"


def test_set_float_var_passthrough():
    assert set_float_var(var=60.0) == 60.0
