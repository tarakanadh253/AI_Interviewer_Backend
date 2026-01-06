# Generated migration to add access_type field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interview', '0002_change_to_username_password_auth'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='access_type',
            field=models.CharField(
                choices=[('TRIAL', 'Trial - One Free Interview'), ('FULL', 'Full Access - Unlimited Interviews')],
                default='TRIAL',
                max_length=10,
                null=True,  # Allow null temporarily for existing data
                blank=True
            ),
        ),
        # Set default for existing rows
        migrations.RunSQL(
            "UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;",
            reverse_sql=migrations.RunSQL.noop,
        ),
        # Make it non-nullable after setting defaults
        migrations.AlterField(
            model_name='userprofile',
            name='access_type',
            field=models.CharField(
                choices=[('TRIAL', 'Trial - One Free Interview'), ('FULL', 'Full Access - Unlimited Interviews')],
                default='TRIAL',
                max_length=10
            ),
        ),
    ]
