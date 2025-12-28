from django.db import migrations
from django.contrib.auth.hashers import make_password

def create_admin_user(apps, schema_editor):
    UserProfile = apps.get_model('interview', 'UserProfile')
    
    # Check if admin user exists, if not create it
    if not UserProfile.objects.filter(username='admin@ohg').exists():
        UserProfile.objects.create(
            username='admin@ohg',
            password=make_password('ohg@365'),
            email='admin@ohg.com',
            name='Admin User',
            is_active=True,
            access_type='FULL' # Admin gets full access
        )

def remove_admin_user(apps, schema_editor):
    UserProfile = apps.get_model('interview', 'UserProfile')
    UserProfile.objects.filter(username='admin@ohg').delete()

class Migration(migrations.Migration):

    dependencies = [
        ('interview', '0004_add_source_type_and_reference_links'),
    ]

    operations = [
        migrations.RunPython(create_admin_user, remove_admin_user),
    ]
