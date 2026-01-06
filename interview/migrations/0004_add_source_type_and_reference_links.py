# Generated manually to fix missing columns
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interview', '0003_add_access_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='source_type',
            field=models.CharField(
                choices=[('MANUAL', 'Manually Defined - Admin enters Q&A directly'), ('LINK', 'From External Links - Questions and answers from provided URLs')],
                default='MANUAL',
                help_text='Choose how to define this question: manually enter Q&A or use external links',
                max_length=10
            ),
        ),
        migrations.AddField(
            model_name='question',
            name='reference_links',
            field=models.TextField(
                blank=True,
                help_text='Enter one URL per line. Required if source_type is LINK. These links contain the questions and answers for the interview.',
                null=True
            ),
        ),
    ]
