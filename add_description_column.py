#!/usr/bin/env python3
"""
Script to add the description column to the filter_states table in Supabase.
Run this once to update the database schema.
"""

def add_description_column():
    """Display the SQL needed to add the description column"""

    print("To fix the filter saving issue, you need to add the 'description' column to the filter_states table.")
    print()
    print("Please run this SQL command in your Supabase SQL Editor (found in your Supabase dashboard):")
    print()
    print("ALTER TABLE filter_states ADD COLUMN IF NOT EXISTS description TEXT;")
    print()
    print("This will add the missing description column that the save_filter_state function expects.")
    print("After running this SQL, filter saving should work correctly in your deployed app.")

if __name__ == "__main__":
    add_description_column()