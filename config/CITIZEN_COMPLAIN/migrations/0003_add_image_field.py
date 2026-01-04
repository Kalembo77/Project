# Auto migration to add image field to RegisteredUser
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('CITIZEN_COMPLAIN', '0002_create_complaint'),
    ]

    operations = [
        migrations.AddField(
            model_name='registereduser',
            name='image',
            field=models.ImageField(upload_to='profile_images/', null=True, blank=True),
        ),
    ]
