# Generated by Django 4.0.1 on 2022-01-14 00:19

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ml_model', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='guest',
            name='user_id',
            field=models.ForeignKey(db_column='user_id', default=9, on_delete=django.db.models.deletion.DO_NOTHING, to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='devicecameraconnectstatus',
            name='modified_time',
            field=models.PositiveBigIntegerField(default=1642087149),
        ),
        migrations.AlterField(
            model_name='devicecameraevent',
            name='created_timestamp',
            field=models.PositiveBigIntegerField(default=1642087149),
        ),
    ]
