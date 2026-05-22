from ecoscope.platform.tasks.results import OutputFiles, gather_output_files


def test_gather_output_files_round_trip() -> None:
    composite_filter = (("region", "=", "north"),)
    files: list = ["a.html", [(composite_filter, "b.html")]]

    result = gather_output_files(files=files)

    assert isinstance(result, OutputFiles)
    assert result.files == files
