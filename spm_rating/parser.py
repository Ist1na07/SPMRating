"""
SPM Rating — .osu file parser.

Refactored from sunnyrework/osu_file_parser.py.
Parses osu!mania beatmap files and returns structured note data.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class NoteData:
    """Structured parsed beatmap data."""
    column_count: int = 0
    columns: list = field(default_factory=list)
    note_starts: list = field(default_factory=list)
    note_ends: list = field(default_factory=list)
    note_types: list = field(default_factory=list)
    od: float = -1.0
    file_path: str = ""
    metadata: dict = field(default_factory=dict)


def _str_to_int(s):
    """Convert string to int (handle float strings like '6.0')."""
    return int(float(s))


# Column remapping tables: non-7K modes → 7K physical column indices (0-indexed)
_COLUMN_REMAP = {
    4: {0: 1, 1: 2, 2: 4, 3: 5},       # 4K → 7K columns 2356
    5: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},  # 5K → 7K columns 23456
    6: {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6},  # 6K → 7K columns 123567
}


def _build_remap_table(original_k):
    """Build a column remap array for the given key count.

    Returns an array where arr[original_col] = 7k_col, or None if no remap needed.
    """
    return _COLUMN_REMAP.get(original_k, None)


class Parser:
    """Parser for osu! .osu beatmap files.

    Non-7K maps (4K/5K/6K) have their columns remapped to 7K physical
    column indices, so the rating pipeline always operates in 7K space:
      4K → 7K columns 2,3,5,6 (indices 1,2,4,5)
      5K → 7K columns 2,3,4,5,6 (indices 1,2,3,4,5)
      6K → 7K columns 1,2,3,5,6,7 (indices 0,1,2,4,5,6)
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = NoteData()
        self.data.file_path = file_path
        self._original_column_count = 7
        self._remap = None

    def process(self):
        """Parse the .osu file and populate self.data."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            try:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Read Metadata section
                    if stripped == "[Metadata]":
                        self._read_metadata(f)

                    # Read OverallDifficulty
                    if stripped.startswith("OverallDifficulty"):
                        self.data.od = self._read_od(stripped)

                    # Read CircleSize (key count for mania)
                    if stripped.startswith("CircleSize"):
                        self.data.column_count = self._read_circle_size(stripped)

                    # Read HitObjects
                    if stripped == "[HitObjects]":
                        self._read_notes(f)

            except StopIteration:
                pass

    def _read_metadata(self, f):
        """Read metadata lines until we find the next section."""
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                return
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                self.data.metadata[key.strip()] = val.strip()

    def _read_od(self, line):
        """Parse OverallDifficulty value."""
        try:
            pos = line.index(':')
            return float(line[pos+1:].strip())
        except (ValueError, IndexError):
            return -1.0

    def _read_circle_size(self, line):
        """Parse CircleSize (column count in mania mode).

        Stores the original key count and builds a column remap table
        for non-7K modes. After parsing notes, column_count is set to 7.
        """
        try:
            pos = line.index(':')
            val = line[pos+1:].strip()
            if val == '0':
                self._original_column_count = 10
                self.data.column_count = 10
                return 10
            k = _str_to_int(val)
            self._original_column_count = k
            self._remap = _build_remap_table(k)
            # Set column_count to 7 for 4K/5K/6K (operate in 7K space)
            self.data.column_count = 7 if k in (4, 5, 6) else k
            return self.data.column_count
        except (ValueError, IndexError):
            return -1

    def _read_notes(self, f):
        """Parse all hit objects."""
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("["):
                return
            self._parse_hit_object(stripped)

    def _parse_hit_object(self, obj_line):
        """Parse a single hit object line.

        Format: x,y,time,type,hitSound,objectParams,hitSample
        For mania: x determines column, hold notes have endTime in objectParams.

        Non-7K maps (4K/5K/6K): X is first mapped to an original column
        (0..K-1) using the original key-count's column width, then remapped
        to the corresponding 7K physical column index.
        """
        params = obj_line.split(",")
        if len(params) < 4:
            return

        # Column from x-coordinate using ORIGINAL key count
        x = int(params[0])
        original_k = self._original_column_count
        column_width = 512.0 / original_k
        column = min(int(x / column_width), original_k - 1)

        # Remap to 7K column space for 4K/5K/6K
        if self._remap is not None:
            column = self._remap.get(column, column)

        # Note start time
        note_start = int(params[2])

        # Note type (bit 7 = hold/LN)
        note_type = int(params[3])

        # Note end time (hold notes only, from objectParams)
        # format of params[5]: endTime:...
        note_end = 0
        if len(params) >= 6:
            last_chunk = params[5].split(":")
            try:
                note_end = int(last_chunk[0])
            except (ValueError, IndexError):
                note_end = 0

        self.data.columns.append(column)
        self.data.note_starts.append(note_start)
        self.data.note_ends.append(note_end)
        self.data.note_types.append(note_type)

    def get_parsed_data(self):
        """Return parsed data as a list matching the SunnyRework format.

        Returns:
            [column_count, columns, note_starts, note_ends, note_types, od]
        """
        return [
            self.data.column_count,
            self.data.columns,
            self.data.note_starts,
            self.data.note_ends,
            self.data.note_types,
            self.data.od,
        ]


def parse_file(file_path):
    """Convenience function to parse a .osu file and return raw data."""
    p = Parser(file_path)
    p.process()
    return p.get_parsed_data()
