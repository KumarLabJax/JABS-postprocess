import json
import pytest
from jabs_postprocess.cli.utils import load_json


class TestLoadJsonNoneInput:
    """Test load_json with None input."""

    def test_none_returns_none(self):
        """None input should return None."""
        assert load_json(None) is None


class TestLoadJsonPathInput:
    """Test load_json with Path objects."""

    def test_path_valid_json_dict(self, tmp_path):
        """Path to file with valid JSON dict should return dict."""
        json_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        json_file.write_text(json.dumps(test_data))

        result = load_json(json_file)
        assert result == test_data

    def test_path_valid_json_list(self, tmp_path):
        """Path to file with valid JSON list should return list."""
        json_file = tmp_path / "test.json"
        test_data = [1, 2, 3, "four"]
        json_file.write_text(json.dumps(test_data))

        result = load_json(json_file)
        assert result == test_data

    def test_path_nonexistent_file(self, tmp_path):
        """Path to non-existent file should raise ValueError."""
        json_file = tmp_path / "nonexistent.json"

        with pytest.raises(ValueError, match="File not found"):
            load_json(json_file)

    def test_path_invalid_json(self, tmp_path):
        """Path to file with invalid JSON should raise ValueError."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{ invalid json content")

        with pytest.raises(ValueError, match="Invalid JSON in file"):
            load_json(json_file)

    def test_path_empty_file(self, tmp_path):
        """Path to empty file should raise ValueError."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("")

        with pytest.raises(ValueError, match="Invalid JSON in file"):
            load_json(json_file)


class TestLoadJsonStringAsJson:
    """Test load_json with string containing JSON content."""

    @pytest.mark.parametrize(
        "json_str,expected",
        [
            ('{"key": "value"}', {"key": "value"}),
            ('{"nested": {"data": [1, 2, 3]}}', {"nested": {"data": [1, 2, 3]}}),
            ("[1, 2, 3]", [1, 2, 3]),
            ('["a", "b", "c"]', ["a", "b", "c"]),
            ('"simple string"', "simple string"),
            ("true", True),
            ("false", False),
            ("null", None),
            ("42", 42),
            ("3.14", 3.14),
        ],
    )
    def test_valid_json_strings(self, json_str, expected):
        """Various valid JSON strings should be parsed correctly."""
        result = load_json(json_str)
        assert result == expected

    @pytest.mark.parametrize(
        "json_str",
        [
            '  {"key": "value"}  ',  # with whitespace
            '\n{\n  "key": "value"\n}\n',  # with newlines
            '\t["a", "b"]\t',  # with tabs
        ],
    )
    def test_json_with_whitespace(self, json_str):
        """JSON strings with surrounding whitespace should parse correctly."""
        result = load_json(json_str)
        assert result is not None

    def test_complex_nested_json(self):
        """Complex nested JSON should parse correctly."""
        json_str = json.dumps(
            {
                "users": [
                    {"name": "Alice", "age": 30, "active": True},
                    {"name": "Bob", "age": 25, "active": False},
                ],
                "meta": {"version": 1, "timestamp": None},
            }
        )
        result = load_json(json_str)
        assert result["users"][0]["name"] == "Alice"
        assert result["meta"]["timestamp"] is None


class TestLoadJsonStringAsFilePath:
    """Test load_json with string as file path."""

    def test_string_path_to_valid_json(self, tmp_path):
        """String path to valid JSON file should work."""
        json_file = tmp_path / "test.json"
        test_data = {"from": "file"}
        json_file.write_text(json.dumps(test_data))

        result = load_json(str(json_file))
        assert result == test_data

    def test_string_path_nonexistent(self):
        """String that looks like path but file doesn't exist should raise ValueError."""
        # Use a path that doesn't start with JSON chars
        with pytest.raises(
            ValueError, match="neither a valid file path nor valid JSON"
        ):
            load_json("nonexistent_file.json")

    def test_string_path_with_invalid_json(self, tmp_path):
        """String path to file with invalid JSON should raise ValueError."""
        json_file = tmp_path / "bad.json"
        json_file.write_text("not valid json")

        with pytest.raises(ValueError, match="Invalid JSON in file"):
            load_json(str(json_file))


class TestLoadJsonAmbiguousCases:
    """Test load_json with ambiguous strings that could be JSON or paths."""

    def test_json_like_string_parsed_as_json_first(self):
        """String starting with { should be tried as JSON first."""
        # This looks like JSON and is valid JSON, so it should parse
        result = load_json('{"file.json": "value"}')
        assert result == {"file.json": "value"}

    def test_invalid_json_fallback_to_file(self, tmp_path):
        """Invalid JSON string might fallback to file path."""
        # Create a file with a name that could be confused with JSON
        json_file = tmp_path / '{"incomplete"'
        test_data = {"actual": "content"}
        json_file.write_text(json.dumps(test_data))

        # This should fail JSON parsing and try as file
        with pytest.raises(ValueError):
            # The string starts with { so it's tried as JSON first
            # When that fails, it tries as file path, which also fails (file doesn't exist)
            load_json('{"incomplete"')

    def test_file_path_not_starting_with_json_char(self, tmp_path):
        """File path that doesn't look like JSON should try file first."""
        json_file = tmp_path / "data.json"
        test_data = {"data": "value"}
        json_file.write_text(json.dumps(test_data))

        result = load_json(str(json_file))
        assert result == test_data


class TestLoadJsonErrorCases:
    """Test load_json error handling."""

    def test_invalid_type_int(self):
        """Integer input should raise TypeError."""
        with pytest.raises(TypeError, match="source must be Path, str, or None"):
            load_json(42)  # type: ignore

    def test_invalid_type_list(self):
        """List input should raise TypeError."""
        with pytest.raises(TypeError, match="source must be Path, str, or None"):
            load_json([1, 2, 3])  # type: ignore

    def test_invalid_type_dict(self):
        """Dict input should raise TypeError."""
        with pytest.raises(TypeError, match="source must be Path, str, or None"):
            load_json({"key": "value"})  # type: ignore

    def test_invalid_json_string_not_file(self):
        """Invalid JSON that's also not a file should raise ValueError."""
        with pytest.raises(
            ValueError, match="neither a valid file path nor valid JSON"
        ):
            load_json("definitely not json or a file path")

    def test_truncated_error_message_for_long_json(self):
        """Long invalid JSON should have truncated error message."""
        long_invalid = "{" + "x" * 200
        with pytest.raises(ValueError, match=r"Invalid JSON.*\.\.\.$"):
            load_json(long_invalid)


class TestLoadJsonEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            load_json("")

    def test_whitespace_only_string(self):
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError):
            load_json("   \n\t  ")

    def test_json_with_escaped_quotes(self):
        """JSON with escaped quotes should parse correctly."""
        json_str = '{"quote": "He said \\"hello\\""}'
        result = load_json(json_str)
        assert result["quote"] == 'He said "hello"'

    def test_deeply_nested_json(self):
        """Deeply nested JSON should parse correctly."""
        nested = {"level": 1}
        current = nested
        for i in range(2, 20):
            current["nested"] = {"level": i}
            current = current["nested"]

        json_str = json.dumps(nested)
        result = load_json(json_str)
        assert result["level"] == 1
        assert result["nested"]["nested"]["level"] == 3


class TestLoadJsonReturnTypes:
    """Test that load_json returns correct types."""

    def test_returns_dict_type(self):
        """Dict JSON should return dict type."""
        result = load_json('{"key": "value"}')
        assert isinstance(result, dict)

    def test_returns_list_type(self):
        """List JSON should return list type."""
        result = load_json("[1, 2, 3]")
        assert isinstance(result, list)

    def test_returns_none_type(self):
        """null JSON should return None."""
        result = load_json("null")
        assert result is None

    def test_returns_bool_type(self):
        """Boolean JSON should return bool."""
        result = load_json("true")
        assert isinstance(result, bool)
        assert result is True

    def test_returns_int_type(self):
        """Integer JSON should return int."""
        result = load_json("42")
        assert isinstance(result, int)
        assert result == 42

    def test_returns_float_type(self):
        """Float JSON should return float."""
        result = load_json("3.14")
        assert isinstance(result, float)
        assert result == 3.14

    def test_returns_string_type(self):
        """String JSON should return str."""
        result = load_json('"hello"')
        assert isinstance(result, str)
        assert result == "hello"
