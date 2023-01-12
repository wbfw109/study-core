import datetime
from django.db import models
from django.utils import timezone
from ml_model.model.base import User


class DeviceCube(models.Model):
    id = models.AutoField(primary_key=True)
    user: User = models.ForeignKey(
        User, on_delete=models.DO_NOTHING, db_column="user_id", null=True
    )
    serial_number: str = models.CharField(max_length=255)
    firmware: str = models.CharField(max_length=255, unique=True)
    name: str = models.CharField(max_length=50)
    is_deleted: bool = models.BooleanField(blank=True, default=False)
    created_datetime: datetime.datetime = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "device_cube"


class Guest(models.Model):
    id: int = models.AutoField(primary_key=True)
    user: User = models.ForeignKey(
        User, on_delete=models.DO_NOTHING, db_column="user_id"
    )
    phone_number: str = models.CharField(max_length=15)
    device_cubes: list[DeviceCube] = models.ManyToManyField(
        DeviceCube, through="DeviceCubeAlertsGuest"
    )
    created_datetime: datetime = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "guest"


class DeviceCubeAlertsGuest(models.Model):
    id: int = models.AutoField(primary_key=True)
    device_cube: DeviceCube = models.ForeignKey(DeviceCube, on_delete=models.CASCADE)
    guest: Guest = models.ForeignKey(Guest, on_delete=models.CASCADE)

    class Meta:
        db_table = "device_cube_alerts_guest"
