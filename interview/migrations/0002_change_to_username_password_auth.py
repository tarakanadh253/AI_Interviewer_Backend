# Generated migration to change from google_id to username/password authentication
# SQLite doesn't support DROP COLUMN, so we recreate the table

from django.db import migrations, models


def recreate_userprofile_table(apps, schema_editor):
    """Recreate user_profiles table with new schema"""
    db_alias = schema_editor.connection.alias
    
    # For SQLite, we need to recreate the table
    with schema_editor.connection.cursor() as cursor:
        # Create new table with correct schema
        cursor.execute("""
            CREATE TABLE user_profiles_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(150) UNIQUE NOT NULL,
                password VARCHAR(128) NOT NULL,
                email VARCHAR(254) NOT NULL,
                name VARCHAR(255),
                is_active BOOLEAN NOT NULL DEFAULT 1,
                has_used_trial BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
        """)
        
        # Create index
        cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles_new(username);")
        
        # Drop old table
        cursor.execute("DROP TABLE IF EXISTS user_profiles;")
        
        # Rename new table
        cursor.execute("ALTER TABLE user_profiles_new RENAME TO user_profiles;")


def reverse_recreate_userprofile_table(apps, schema_editor):
    """Reverse migration - recreate with google_id"""
    with schema_editor.connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE user_profiles_old (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                google_id VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(254) NOT NULL,
                name VARCHAR(255),
                has_used_trial BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
        """)
        cursor.execute("CREATE UNIQUE INDEX user_profiles_google_id_idx ON user_profiles_old(google_id);")
        cursor.execute("DROP TABLE IF EXISTS user_profiles;")
        cursor.execute("ALTER TABLE user_profiles_old RENAME TO user_profiles;")


class Migration(migrations.Migration):

    dependencies = [
        ('interview', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(recreate_userprofile_table, reverse_recreate_userprofile_table),
    ]
