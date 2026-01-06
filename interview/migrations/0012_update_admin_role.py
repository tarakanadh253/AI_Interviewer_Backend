from django.db import migrations

def update_admin_role(apps, schema_editor):
    UserProfile = apps.get_model('interview', 'UserProfile')
    # Update regular admin
    UserProfile.objects.filter(username='admin@ohg').update(role='ADMIN')
    # Update seeded admin if separate
    UserProfile.objects.filter(username='admin').update(role='ADMIN')

class Migration(migrations.Migration):

    dependencies = [
        ('interview', '0011_userprofile_enrolled_course_userprofile_student_id'),
    ]

    operations = [
        migrations.RunPython(update_admin_role),
    ]
