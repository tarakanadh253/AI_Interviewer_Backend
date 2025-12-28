# Generated migration to seed initial data

from django.db import migrations

def seed_data(apps, schema_editor):
    """
    Skipping seed data as per user request.
    Data should be manually entered by admin.
    """
    pass

def remove_seeded_data(apps, schema_editor):
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('interview', '0005_create_admin_user'),
    ]

    operations = [
        migrations.RunPython(seed_data, remove_seeded_data),
    ]
