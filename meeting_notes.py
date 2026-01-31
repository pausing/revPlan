"""
Meeting Notes Module for Engineering Plan Review

This module handles the storage and retrieval of meeting notes organized by date.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional


NOTES_FILE = 'meeting_notes.json'


def load_notes() -> Dict[str, str]:
    """
    Load meeting notes from file.
    
    Returns:
        Dictionary with date strings as keys and notes as values
    """
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading notes: {e}")
            return {}
    return {}


def save_notes(notes: Dict[str, str]):
    """
    Save meeting notes to file.
    
    Args:
        notes: Dictionary with date strings as keys and notes as values
    """
    try:
        with open(NOTES_FILE, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving notes: {e}")


def get_note_for_date(date: datetime) -> str:
    """
    Get note for a specific date.
    
    Args:
        date: datetime object
        
    Returns:
        Note text for the date, or empty string if not found
    """
    notes = load_notes()
    date_str = date.strftime('%Y-%m-%d')
    return notes.get(date_str, '')


def save_note_for_date(date: datetime, note: str):
    """
    Save note for a specific date.
    
    Args:
        date: datetime object
        note: Note text to save
    """
    notes = load_notes()
    date_str = date.strftime('%Y-%m-%d')
    notes[date_str] = note
    save_notes(notes)


def get_all_notes() -> Dict[str, str]:
    """
    Get all notes sorted by date (most recent first).
    
    Returns:
        Dictionary with date strings as keys and notes as values, sorted by date
    """
    notes = load_notes()
    # Sort by date (most recent first)
    sorted_notes = dict(sorted(notes.items(), key=lambda x: x[0], reverse=True))
    return sorted_notes


def delete_note_for_date(date: datetime):
    """
    Delete note for a specific date.
    
    Args:
        date: datetime object
    """
    notes = load_notes()
    date_str = date.strftime('%Y-%m-%d')
    if date_str in notes:
        del notes[date_str]
        save_notes(notes)
