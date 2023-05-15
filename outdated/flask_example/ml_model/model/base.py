from __future__ import annotations
from django.contrib.auth.models import User
from django.db import models
from typing import Optional
import dataclasses
import time

class UserProfile(models.Model):
    id: int = models.AutoField(primary_key=True)
    user: User = models.OneToOneField(User, on_delete=models.DO_NOTHING)
    # created_datetime = models.DateTimeField(default=timezone.now)
    nickname: str = models.CharField(max_length=100, blank=True, default="")
    phone_number: str = models.CharField(max_length=15)
    recent_camera_query: str = models.TextField(blank=True, default="")
    class Meta:
        db_table = "user_profile"



class ThreatEvent(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)

    class Meta:
        db_table = u"threat_event"


@dataclasses.dataclass()
class ConnectStatus:
    """It used to check the health of connect status."""

    modified_time: Optional[float] = dataclasses.field(default=time.time())
    is_connected: int = dataclasses.field(default=False)

    def set_connect_status_from_connect_threshold_seconds(
        self, connect_threshold_seconds: int
    ) -> None:
        current_time = time.time()
        if (current_time - self.modified_time) < connect_threshold_seconds:
            self.connect_status = ConnectStatus(
                modified_time=current_time, is_connected=True
            )
        else:
            self.connect_status.is_connected = False

    def set_connect_status(self, connect_status: ConnectStatus) -> None:
        """
        - if a attribute of connect_status is None, it considered as not connected.
            so connect_status.modifed_time must not be updated.
        """
        if connect_status.modified_time is None or connect_status.is_connected is None:
            self.connect_status.is_connected = False
        else:
            self.connect_status = connect_status
