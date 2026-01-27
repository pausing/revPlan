"""
Meeting Notes Module
Handles storage and retrieval of meeting notes by date.
"""
import json
import os
from datetime import datetime

# File to store notes
NOTES_FILE = 'meeting_notes.json'

def _load_notes_file():
    """Load notes from JSON file. Returns dict with date strings as keys."""
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _save_notes_file(notes_dict):
    """Save notes dict to JSON file."""
    with open(NOTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(notes_dict, f, indent=2, ensure_ascii=False)

def _date_to_key(date_obj):
    """Convert datetime/date object to string key (YYYY-MM-DD)."""
    if isinstance(date_obj, datetime):
        return date_obj.date().strftime('%Y-%m-%d')
    elif hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%Y-%m-%d')
    else:
        return str(date_obj)

def load_notes():
    """Load all notes. Returns dict with date strings as keys."""
    return _load_notes_file()

def get_note_for_date(date_obj):
    """Get note content for a specific date. Returns empty string if not found."""
    notes = _load_notes_file()
    date_key = _date_to_key(date_obj)
    return notes.get(date_key, "")

def save_note_for_date(date_obj, note_content):
    """Save note content for a specific date."""
    notes = _load_notes_file()
    date_key = _date_to_key(date_obj)
    notes[date_key] = note_content
    _save_notes_file(notes)

def get_all_notes():
    """Get all notes as a dict with date strings (YYYY-MM-DD) as keys."""
    return _load_notes_file()

def delete_note_for_date(date_obj):
    """Delete note for a specific date."""
    notes = _load_notes_file()
    date_key = _date_to_key(date_obj)
    if date_key in notes:
        del notes[date_key]
        _save_notes_file(notes)
        return True
    return False
